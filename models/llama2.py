import torch
import json
import argparse
import sys
import os

from string import Template
from typing import TextIO

from hurry.filesize import size as size_bytes
from huggingface_hub import login
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

def safe_open_w(path: str) -> TextIO:
    """
    Open "path" for writing, creating any parent directories as needed.

    Args
        path: path to file, in string format

    Return
        file: TextIO object
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def check_memory_footprints():
    print(size_bytes(LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", \
                                                      device_map="auto", load_in_8bit=True).get_memory_footprint()))
    
    print(size_bytes(LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", \
                                                      device_map="auto", load_in_4bit=True).get_memory_footprint()))
    
    print(size_bytes(LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", \
                                                      device_map="auto", load_in_8bit=True).get_memory_footprint()))
    
    print(size_bytes(LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", \
                                                      device_map="auto", load_in_4bit=True).get_memory_footprint()))

class LLaMAInferencer():
    def __init__(self, base_model : str = "meta-llama/Llama-2-13b-chat-hf" , max_tokens : str = 4096, device : str = 'cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.get_model(base_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)

        #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # the above adds +1 to the vocab count causing a mismatch. below solves:
        self.tokenizer.pad_token='[PAD]'

    @staticmethod
    def get_model(base_model : str = "meta-llama/Llama-2-13b-chat-hf", device : str = 'cuda'):
        return LlamaForCausalLM.from_pretrained(base_model, device_map="auto", load_in_8bit=True)
    
    @staticmethod
    def get_tokenizer(base_model : str = "meta-llama/Llama-2-13b-chat-hf"):
        return LlamaTokenizer.from_pretrained(base_model)
    
    def build_simple_prompt(self, prompt : str, queries : list[str]) -> list[str]:
        return [Template(prompt).substitute(query=query) for query in queries]

    def tokenize_inference(self, texts : list[str]) -> dict[str,torch.Tensor]:
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens).to("cuda")
        # Force end of prompt to not be truncated
        # for example in tokenized['input_ids']:
        #    # if it was padded, there is no need
        #    if 0 not in example:
        #        example[-4:] = torch.LongTensor([830, 6591, 13641, 29901])
        return tokenized
    
    # documents: list of strings; each string a document.
    def inference(self, prompt : str, queries : list[str] = [], batchsize : int = 1, generation_config : GenerationConfig = None) -> list[str]:
        model_eval = self.model.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        def batch(X, batch_size : int = 1) -> list[list[str]]:
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        outputs = []
        for sample in batch(queries, batchsize):
            inputs = self.tokenize_inference(self.build_simple_prompt(prompt, sample))
            sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
            outputs.extend(sample_outputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def build_trial_gpt_prompt(self, prompt : str, query_patient : str, clinical_trial : str) -> str:
        return None
    
    # documents: list of strings; each string a document.
    def trialgpt_eligibility_inference(self, prompt_file : dict[str], patient_queries : list[str], clinical_trials : list[str], generation_config : GenerationConfig = None) -> list[str]:
        model_eval = self.model.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        outputs_by_patient_query = {}
        outputs = []

        base_prompt = Template(prompt_file["full_criterion-level_eligibility_prompt"]).substitute(eligibility = prompt_file["eligibility_prompt"], in_context_example = prompt_file["in_context_eligibility-example"])

        print(prompt_file)
        print(patient_queries)
        print(clinical_trials)



        #for sample in batch(queries, batchsize):
        #    inputs = self.tokenize_inference(prompt, sample)
        #    sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
        #    outputs.extend(sample_outputs)


        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()

    # model used, and max lenght of input
    parser.add_argument('--base_model', type=str, help='base model to use', default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument('--max_tokens', type=int, help='max tokens to use', default=4096)

    # Path to prompt files, queries and qrels files
    parser.add_argument('--prompt_file', type=str, help='path to queries file', default="../prompts/TrialGPT.json")
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/TREC2023/custom_queries2023.json")
    parser.add_argument('--trials', type=str, help='path to queries file', default="../queries/TREC2023/custom_queries2023.json")

    # Output options and directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/TREC2023/ranking/")
    args = parser.parse_args()

    # Creating our inferencer object
    llama = LLaMAInferencer(base_model=args.base_model, max_tokens=args.max_tokens)

    # Loading files
    prompt_file = json.load(open(args.prompt_file))
    queries = json.load(open(args.queries))
    trials = json.load(open(args.trials))
    
    # Running inference
    llama.trialgpt_eligibility_inference(prompt_file, queries, trials)

if __name__ == '__main__':
    main()