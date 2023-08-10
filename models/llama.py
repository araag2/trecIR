import torch
import sys

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

class LLaMAQueryGenerator():
    def __init__(self, base_model : str = "yahma/llama-7b-hf" , max_tokens : str = 2048, device : str = 'cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = self.get_model_train(base_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)
        #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # the above adds +1 to the vocab count causing a mismatch. below solves:
        self.tokenizer.pad_token='[PAD]'

    @staticmethod
    def get_model_train(base_model : str = "yahma/llama-7b-hf", device : str = 'cuda'):
        return LlamaForCausalLM.from_pretrained(base_model).to(device)
    
    @staticmethod
    def get_tokenizer(base_model : str = "yahma/llama-7b-hf"):
        return LlamaTokenizer.from_pretrained(base_model)
    
    def tokenize_train(self, batch : list[dict[str,str]]) -> dict[str,torch.Tensor]:
        texts = []
        labels = []
        for example in batch:
            document = example['doc']
            query = example['query']
            texts.append(f'Generate query: {document}. Query:')
            labels.append(query)
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        tokenized['labels'] = self.tokenizer(labels, return_tensors='pt', padding=True, truncation='longest_first')['input_ids']
        
        # Force "Query:" to be at the end of the prompt, if it gets truncated. 
        # [13641, 29901] is the tokenized version of "Query:"
        for example in tokenized['input_ids']:
            example[-2:] = torch.LongTensor([13641, 29901])
        for example in tokenized['attention_mask']:
            example[-2:] = torch.LongTensor([1, 1])

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    
    def tokenize_inference(self, batch : list[str]) -> dict[str,torch.Tensor]:
        texts = []
        for example in batch:
            document = example
            
            #texts.append(f'Given a query: {document}. Summarize the query:')
            texts.append(f'Generate query: {document}. Relevant Query:')
            #texts.append(document)
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        
        # Force "Relevant Query:" to be at the end of the prompt, if it gets truncated. 
        # [830, 6591, 13641, 29901] is the tokenized version of "Relevant Query:"
        for example in tokenized['input_ids']:
            # if it was padded, there is no need
            if 0 not in example:
                example[-4:] = torch.LongTensor([830, 6591, 13641, 29901])

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized

    def few_shot_prompting(self, documents : list[str] = [], batchsize : int = 300, generation_config : GenerationConfig=None,examples : list[str] = [], prompt_prefix : str = '', doc_prefix : str = 'Example Doc:', query_prefix: str ='\nRelevant Query:') -> list[str]:
        # examples = [{'doc':doc, 'query':query}, ...]
        model_eval = self.model_train.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config
        
        def batch(X, batch_size : int = 1) -> list[list[str]]:
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        outputs = []
        for sample in batch(documents, batchsize):
            prompt = self.build_few_shot_prompt(sample, examples, prompt_prefix, doc_prefix, query_prefix)
            inputs = self.tokenize_inference(prompt)
            sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
            outputs.extend(sample_outputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def build_few_shot_prompt(self, docs : list[str] = [], examples : list[str] = [], prompt_prefix : str = '', doc_prefix : str = '' , query_prefix  : str = '') -> list[str]:
        prompts = []
        for doc in docs:
            prompt = prompt_prefix
            for example in examples:
                prompt += doc_prefix + example['doc'] + query_prefix + example['query']
            prompt +=  'Example Doc:' + doc + '\nRelevant Query:'
            prompts.append(prompt)
        return prompt
    
    # documents: list of strings; each string a document.
    def inference(self, documents : list[str] = [], batchsize : int = 300, generation_config : GenerationConfig=None) -> list[str]:
        model_eval = self.model_train.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        def batch(X, batch_size : int = 1) -> list[list[str]]:
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        outputs = []
        for sample in batch(documents, batchsize):
            inputs = self.tokenize_inference(sample)
            sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
            outputs.extend(sample_outputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)