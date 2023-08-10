from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch


@torch.no_grad()
def greedy_decode(model,
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True):
    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True)
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids


# This class was adapted from MonoT5
class T5QueryFilter():
    def __init__(self, 
                 model_path: str,
                 use_amp = True,
                 token_false = '▁false',
                 token_true  = '▁true',
                 batch_size=600):
        self.model_train = self.get_model(model_path)
        self.tokenizer = self.get_tokenizer(model_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(self.tokenizer, token_false, token_true)
        self.model_path = model_path
        self.device = next(self.model_train.parameters(), None).device
        self.use_amp = use_amp
        self.batch_size = batch_size
        

    @staticmethod
    def get_model(model_path: str, *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(model_path, *args, **kwargs).to(device)
    
    @staticmethod
    def get_tokenizer(model_path):
        return AutoTokenizer.from_pretrained(model_path)

    @staticmethod
    def get_prediction_tokens(tokenizer, token_false, token_true):
        if (token_false and token_true):
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id
        
    def tokenize(self, batch):        
        texts = []
        for example in batch:
            document = example[1]
            query = example[0]
            texts.append(f'Query: {query}. Document: {document}. Relevant:')
        
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        
        return tokenized
    
    def filter(self, pairs, threshold=0.7, return_scores=False, **kwargs):

        model_eval = self.model_train.eval()
        def batcher(X, batch_size=1):
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        scores = []
        for batch in batcher(pairs, self.batch_size):            
            tokenized = self.tokenize(batch) 
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = tokenized['input_ids'].to(self.device)
                attn_mask = tokenized['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(model_eval,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
                batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
                scores.extend(batch_log_probs)
        
        assert len(scores) == len(pairs) # Sanity check, should be equal

        filter_mask = [True if score > threshold else False for score in scores]

        return (filter_mask, scores if return_scores else None)



# This can actually be used with any Seq2Seq model on hugging face.
# It will read the model type from the base model.

# Still, while this is OK for T5 and LongT5 tokenizers, others may differ.
class T5QueryGenerator():
    def __init__(self, base_model="t5-base", max_tokens=512, device='cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = self.get_model_train(base_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)

    @staticmethod
    def get_model_train(base_model, device):
        return AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    
    @staticmethod
    def get_tokenizer(base_model):
        return AutoTokenizer.from_pretrained(base_model)
    
    def tokenize_train(self, batch):
        texts = []
        labels = []
        for example in batch:
            document = example['doc']
            query = example['query']
            texts.append(f'Generate query: {document}. Query:')
            labels.append(query)
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        tokenized['labels'] = self.tokenizer(labels, return_tensors='pt', padding=True, truncation='longest_first')['input_ids']
        
        # Force "Query:<eos>" to be at the end of the prompt, if it gets truncated.
        for example in tokenized['input_ids']:
            example[-4:] = torch.LongTensor([3, 27569, 10, 1])
        for example in tokenized['attention_mask']:
            example[-4:] = torch.LongTensor([1, 1, 1, 1])

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    
    def tokenize_inference(self, batch):
        texts = []
        for example in batch:
            document = example
            texts.append(f'Generate query: {document}. Query:')
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        
        # Force "Query:<eos>" to be at the end of the prompt, if it gets truncated.
        for example in tokenized['input_ids']:
            example[-4:] = torch.LongTensor([3, 27569, 10, 1])
        for example in tokenized['attention_mask']:
            example[-4:] = torch.LongTensor([1, 1, 1, 1])

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    

    # documents: list of strings; each string a document.
    def inference(self, documents, batchsize=300, generation_config=None):
        model_eval = self.model_train.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        def batch(X, batch_size=1):
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        outputs = []
        for sample in batch(documents, batchsize):
            inputs = self.tokenize_inference(sample)
            sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
            outputs.extend(sample_outputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)