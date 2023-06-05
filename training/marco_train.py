from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, DistributedSampler
import argparse
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from models import (
    LLaMAQueryGenerator,
    T5QueryGenerator,
)
import wandb
import evaluate

class NonShuffleSeq2SeqTrainer(Seq2SeqTrainer):
    # file should be preshuffled
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        # change this for a distributed sampler if training on multi-gpu
        train_sampler = SequentialSampler(self.train_dataset)
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle = False
        )
        
class QueryGenerationDatasetMemory(Dataset):
    # This dataset expects a line by line file with "query\tdocument"
    def __init__(self, filename):
        self._filename = filename
        self._total_data = 0
        self.lines = None
        with open(filename, "r") as file:
            self.lines = [line for line in file]

        self._total_data = int(len(self.lines)-1)

    def __getitem__(self, idx):
        line = self.lines[idx]
        query, doc = line.split("\t")
        
        return {'query': query, 'doc': doc}

    def __len__(self):
        return self._total_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-large', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--model_type", default='t5', type=str, required=False,
                        help="t5 or llama. t5 also works for longt5, just feed in a longt5 base model")
    parser.add_argument("--max_tokens", default=512, type=int, required=False,
                        help="tokenizer max tokens")
    parser.add_argument("--train_pairs_path", default="/public_novasearchdl/jcoelho/msmarco/msmarco_v2_query_generation/marco_squad_train.tsv", type=str, required=False,
                        help="Triples.tsv path")
    parser.add_argument("--eval_pairs_path", default='/public_novasearchdl/jcoelho/msmarco/msmarco_v2_query_generation/dev1.tsv', type=str, required=False,
                        help="Triples.tsv path")
    parser.add_argument("--test_pairs_path", default='/public_novasearchdl/jcoelho/msmarco/msmarco_v2_query_generation/dev2.tsv', type=str, required=False,
                        help="Triples.tsv path")                    
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--warmup_steps", default=250, type=int, required=False,
                        help="Number of warmup steps")
    parser.add_argument("--wandb_run_name", default=None, type=str, required=True,
                        help="WandB run name")
    parser.add_argument("--dataloader_num_workers", default=0, type=int, required=False,
                        help="Num workers for dataloader")

    device = 'cuda'
    torch.manual_seed(123)
    args = parser.parse_args()

    if args.model_type == "t5":
        query_generator = T5QueryGenerator(base_model=args.base_model, max_tokens=args.max_tokens, device=device)
    elif args.model_type == "llama":
        query_generator = LLaMAQueryGenerator(base_model=args.base_model, max_tokens=args.max_tokens, device=device)
    else:
        print("Supported model_types : t5 or llama")
        exit() 

    dataset_train = QueryGenerationDatasetMemory(args.train_pairs_path)
    dataset_eval = QueryGenerationDatasetMemory(args.eval_pairs_path)
    dataset_test = QueryGenerationDatasetMemory(args.test_pairs_path)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    def compute_metrics(eval_preds):        
        metric_bleu = evaluate.load("bleu")
        metric_rouge = evaluate.load("rouge")
        logits, labels = eval_preds

        # Ignore tokens with -100:

        for logit in logits:
            for i in range(len(logit)):
                if logit[i] < 0:
                    logit[i] = 0

        for label in labels:
            for i in range(len(label)):
                if label[i] < 0:
                    label[i] = 0

        # Decode to string:

        str_labels = [query_generator.tokenizer.decode(k) for k in labels]
        str_preds = [query_generator.tokenizer.decode(k) for k in logits]


        bleu = metric_bleu.compute(predictions=str_preds, references=str_labels)['bleu']
        rouge = metric_rouge.compute(predictions=str_preds, references=str_labels)

        reported_metrics = {'bleu': bleu}
        reported_metrics.update(rouge)
        return reported_metrics

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        eval_steps=1500,
        evaluation_strategy = "steps",
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
        dataloader_num_workers = args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer = NonShuffleSeq2SeqTrainer(
        model=query_generator.model_train,
        args=train_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=query_generator.tokenizer,
        data_collator=query_generator.tokenize_train,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()

    predictions = trainer.predict(dataset_test)
    test_metris = predictions[2]
    wandb.log({k.replace("test_", "test/"):v for k,v in test_metris.items()})


if __name__ == '__main__':
    main()
    