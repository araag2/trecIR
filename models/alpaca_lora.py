import torch

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import sys

import torch
from peft import PeftModel


class AlpacaLoRAQueryGenerator():
    def __init__(self, base_model : str = "yahma/llama-7b-hf", alpaca_model: str = 'tloen/alpaca-lora-7b', max_tokens : str = 2048, device : str = 'cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = self.get_model_train(base_model, alpaca_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)
        self.model_train.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model_train.config.bos_token_id = 1
        self.model_train.config.eos_token_id = 2

    @staticmethod
    def get_model_train(base_model : str = "yahma/llama-7b-hf", alpaca_model: str = 'tloen/alpaca-lora-7b', device : str = 'cuda'):
        model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map='auto')
        model = PeftModel.from_pretrained(
            model,
            alpaca_model,
            torch_dtype=torch.float16,
        ).to(device)
        return model
        
    
    @staticmethod
    def get_tokenizer(base_model : str = "yahma/llama-7b-hf"):
        return LlamaTokenizer.from_pretrained(base_model)
    
    def tokenize_inference(self, batch : list[str]) -> dict[str,torch.Tensor]:
        texts = []
        for example in batch:
            document = example
            texts.append(document)
            
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
    
