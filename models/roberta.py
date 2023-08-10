from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch


class RoBERTaQAQueryFilter():
    def __init__(self, 
                 model_path: str = "deepset/roberta-base-squad2",
                 batch_size=600):
        self.model_path = model_path
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.pipeline = pipeline('question-answering', model=model_path, tokenizer=model_path)
        

        
    def filter(self, pairs, threshold=0.7, return_answers=False, **kwargs):

        pipeline_input = [{'question': q, 'context': c} for q,c in pairs]

        output = self.pipeline(pipeline_input, handle_impossible_answer = True, batch_size=self.batch_size)

        if len(pairs) == 1:
            output = [output]

        filter_mask = [True if ans['answer'] != '' and ans['score'] > threshold else False for ans in output]

        return (filter_mask, output if return_answers else None)
    

#>>> from roberta import RoBERTaQAQueryFilter
#>>> model = RoBERTaQAQueryFilter()
#>>> model.filter([('Whats my name?', 'I jave nothing tot do with this')], return_answers=False)
#([False], None)
#>>> model.filter([('Whats my name?', 'I jave nothing tot do with this'), ('Whats my name?', 'My name is Joao')], return_answers=False)
#([False, True], None)