import torch
import json
import argparse
import sys
import os

from tqdm import tqdm
from string import Template
from typing import TextIO

from hurry.filesize import size as size_bytes
from huggingface_hub import login
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, BitsAndBytesConfig

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

        self.tokenizer = self.get_tokenizer(base_model)

        self.model = self.get_model(base_model, self.device)
        self.model.eval()
        print(f"Model loaded to {self.device} with name {base_model}")
        print(size_bytes(self.model.get_memory_footprint()))

        #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # the above adds +1 to the vocab count causing a mismatch. below solves:
        self.tokenizer.pad_token='[PAD]'

    @staticmethod
    def get_model(base_model : str = "meta-llama/Llama-2-13b-chat-hf", device : str = 'cuda'):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return LlamaForCausalLM.from_pretrained(base_model, device_map="auto", quantization_config=bnb_config)
    
    @staticmethod
    def get_tokenizer(base_model : str = "meta-llama/Llama-2-13b-chat-hf"):
        return LlamaTokenizer.from_pretrained(base_model, padding_side='left')
    
    def build_simple_prompt(self, prompt : str, queries : list[str]) -> list[str]:
        return [Template(prompt).substitute(query=queries[patient_id]) for patient_id in queries]

    def tokenize_inference(self, texts : list[str]) -> dict[str,torch.Tensor]:
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens).to("cuda")
        # Force end of prompt to not be truncated
        # for example in tokenized['input_ids']:
        #    # if it was padded, there is no need
        #    if 0 not in example:
        #        example[-4:] = torch.LongTensor([830, 6591, 13641, 29901])
        return tokenized
    
    # documents: list of strings; each string a document.
    def inference(self, prompt : str, queries : list[str] = [], generation_config : GenerationConfig = None) -> list[str]:
        model_eval = self.model.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        inputs = self.tokenize_inference(self.build_simple_prompt(prompt, queries))
        input_lengths = [len(inp) for inp in inputs["input_ids"]]
        outputs = model_eval.generate(**inputs, generation_config=generation_config)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def build_trial_gpt_prompt(self, prompt : str, query_patient : str, trials_corpus : list[str], trials_bm25hits : list[str]) -> str:
        return [Template(prompt).substitute(query_patient=query_patient, clinical_trial=trials_corpus[trial][:6500]) for trial in trials_bm25hits]
    
        # documents: list of strings; each string a document.
    def trialgpt_eligibility_inference(self, 
                                       prompt_file : dict[str], patient_queries : list[str], trials_corpus : list[str], trials_bm25scores : dict[str], output_dir : str, top_k = 1000, batch_size = 5, generation_config : GenerationConfig = None) -> list[str]:
        model = self.model
        generation_config = generation_config if generation_config is not None else model.generation_config

        generation_config.max_new_tokens = 1024

        outputs_by_patient_query = {}

        base_prompt = Template(prompt_file["full_criterion-level_eligibility_prompt"]).safe_substitute(eligibility = prompt_file["eligibility_prompt"], in_context_example = prompt_file["in_context_eligibility-example"])

        def batch(X, batch_size : int = 1) -> list[list[str]]:
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        for patient in tqdm(patient_queries):
            outputs_by_patient_query[patient] = {"clinical_trials" : trials_bm25scores[patient]['docs'][:top_k], "criterion_outputs" : [], "inclusion_scores" : []}

            patient_prompts = self.build_trial_gpt_prompt(base_prompt, patient_queries[patient], trials_corpus, trials_bm25scores[patient]['docs'][:top_k])

            generated_tokens = []

            for sample in tqdm(batch(patient_prompts, batch_size)):
                tokenized_input = self.tokenize_inference(sample)
                generated_tokens.extend(model.generate(**tokenized_input, generation_config=generation_config))
                
            decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for output in decoded_outputs:

                result = output[0].split("Plain JSON output without indent:")

                if len(result) > 1:
                    result = result[1]
                    outputs_by_patient_query[patient]["criterion_outputs"].append(result)

                    included_score = result.count("\"included\"")
                    not_included_score = result.count("\"not included\"")
                    not_relevant_score = result.count("\"no relevant information\"")
                    total_M = max(included_score + not_included_score + not_relevant_score,1)

                    outputs_by_patient_query[patient]["inclusion_scores"].append((included_score/total_M,not_included_score/total_M, not_relevant_score/total_M))

                else:
                    outputs_by_patient_query[patient]["criterion_outputs"].append("")
                    outputs_by_patient_query[patient]["inclusion_scores"].append((0,0,0))

                with safe_open_w(output_dir) as output_f:
                    json.dump(outputs_by_patient_query[patient], output_f, indent=4)

        return outputs_by_patient_query
    
    def trialgpt_trial_relevancy_inference(self, prompt_file : dict[str], patient_queries : list[str], trials_corpus : list[str], trials_bm25scores : dict[str], output_dir : str, generation_config : GenerationConfig = None) -> list[str]:
        model = self.model.eval()
        generation_config = generation_config if generation_config is not None else model.generation_config
        generation_config.max_new_tokens = 4096

        base_prompt = Template(prompt_file["relevancy_prompt"])
        outputs_by_patient_query = {}
        
        for patient in tqdm(patient_queries):
            patient_prompts = self.build_trial_gpt_prompt(base_prompt, patient_queries[patient], trials_corpus, trials_bm25scores[patient]['docs'][:1000])

            tokenized_input = self.tokenize_inference(patient_prompts)

            for tok_inp in tqdm(tokenized_input["input_ids"]):
                decoded_output = self.tokenizer.batch_decode(model.generate(torch.unsqueeze(tok_inp, 0), generation_config=generation_config),skip_special_tokens=True)

                result = decoded_output[0].split("'R=75, E=-50'.")

                if len(result) > 1:
                    outputs_by_patient_query[patient]["relevancy"].append(result[1])

                else:
                    outputs_by_patient_query[patient]["relevancy"].append(result)

            with safe_open_w(output_dir) as output_f:
                json.dump(outputs_by_patient_query, output_f, indent=4)

        return outputs_by_patient_query

def main():
    parser = argparse.ArgumentParser()

    # model used, and max lenght of input
    parser.add_argument('--base_model', type=str, help='base model to use', default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--max_tokens', type=int, help='max tokens to use', default=4096)

    # Path to prompt files, queries and qrels files
    parser.add_argument('--prompt_file', type=str, help='path to prompt file', default="../prompts/TrialGPT.json")
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/TREC2023/queries2023_manual-expand.json")
    parser.add_argument('--trials', type=str, help='path to clinical trials file', default="../datasets/TREC2023/TREC2023_CT-TrialGPT-inclusion.json")
    parser.add_argument('--trials_bm25scores', type=str, help='path to queries file', default="../outputs/TREC2023/ranking/res-free_txt-top-10000-hits.json")
    parser.add_argument('--trials_patient_scores', type=str, help='path to queries file', default="../outputs/TREC2023/prompts/TODO.json")

    # Output options and directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/TREC2023/prompts/")
    args = parser.parse_args()

    # Creating our inferencer object
    llama = LLaMAInferencer(base_model=args.base_model, max_tokens=args.max_tokens)

    print("Torch version:",torch.__version__)

    print("Is CUDA enabled?",torch.cuda.is_available())

    # Loading files
    prompt_file = json.load(open(args.prompt_file))
    queries = json.load(open(args.queries))
    trials_corpus = json.load(open(args.trials))
    trials_bm25scores = json.load(open(args.trials_bm25scores))
    #trials_patient_scores = json.load(open(args.trials_patient_scores))
    
    # Running inference
    #results = llama.inference(Template(prompt_file["expansion_prompt"]).safe_substitute(disease_description = prompt_file["COPD"]), queries)

    results = llama.trialgpt_eligibility_inference(prompt_file, queries, trials_corpus, trials_bm25scores, f'{args.output_dir}{args.queries.split("/")[-1][:-5]}_trialgpt-inclusion.json')

    with safe_open_w(f'{args.output_dir}{args.queries.split("/")[-1][:-5]}_trialgpt-inclusion.json') as output_f:
        json.dump(results, output_f, indent=4)

if __name__ == '__main__':
    main()