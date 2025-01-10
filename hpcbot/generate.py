#hpcbot/generate.py
from .utils import get_chunks, generate_questions, generate_COT_answer, generate_bad_answers, get_final_answer
import json
import random
from openai import OpenAI
import os
    
class QAContextDistractors:
    def __init__(self, model, api_key, base_url, document_dir, out_dir):
        '''
        if using openAI, set base_url = None, api_key = OPENAI_API_KEY
        '''
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.input = document_dir
        self.output = out_dir

    def get_chunks(self):
        return get_chunks(self.input)

    def generate_questions(self, chunk, num=3):
        return generate_questions(self.client, self.model, chunk, num)
    
    def generate_COT_answer(self, question, chunk):
        return generate_COT_answer(self.client, self.model, question, chunk)

    def get_final_answer(self, queries):
        return get_final_answer(queries)
    
    def run(self, num_questions=3, num_distractors=3, include_gold=0.8, stop_early = True):
        dataset = []
        print("getting chunks")
        chunks = self.get_chunks()
        print("generating_QA...")
        for i, (chunk, source) in enumerate(chunks):
            questions = self.generate_questions(chunk, num=num_questions)
            for question in questions:
                data_point = {
                    "id": f"{source}_{i}", 
                    "context": [], 
                    "golden_context": None,
                    "question": question, 
                    "cot_answer": None,
                    "answer": None                   
                    }
                # Add distractor documents
                contexts = random.sample(chunks, num_distractors + 1)
                golden = random.uniform(0, 1) < include_gold
                if golden:
                    contexts[0] = chunk
                random.shuffle(contexts)
                data_point["context"] = [context[0] for context in contexts]
                data_point["golden_context"] = chunk
                # Add answer
                COT_answer = self.generate_COT_answer(question, chunk)
                data_point["cot_answer"] = COT_answer
                data_point["answer"] = self.get_final_answer(COT_answer)
                dataset.append(data_point)
            if stop_early:
                break
        print("copying to: ", self.output)
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        print("Done!")
        return dataset
        
class QAAnswerDistractors:
    def __init__(self, model, api_key, base_url, document_dir, out_dir):
        '''
        if using openAI, set base_url = None, api_key = OPENAI_API_KEY
        '''
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.input = document_dir
        self.output = out_dir

    def get_chunks(self):
        return get_chunks(self.input)

    def generate_questions(self, chunk, num=3):
        return generate_questions(self.client, self.model, chunk, num)
    
    def generate_COT_answer(self, question, chunk):
        return generate_COT_answer(self.client, self.model, question, chunk)

    def get_final_answer(self, queries):
        return get_final_answer(queries)
    
    def generate_bad_answers(self, question, chunk, num_answer):
        return generate_bad_answers(self.client, self.model, question, chunk, num_answer)
    
    def run(self, num_questions = 3, num_answers = 4, stop_early = True):
        dataset = []
        print("getting chunks")
        chunks = self.get_chunks()
        print("generating_QA...")
        for i, (chunk, source) in enumerate(chunks):
            questions = self.generate_questions(chunk, num=num_questions)
            for question in questions:
                data = {
                    "id": f"{source}_{i}",
                    "context": chunk,
                    "question": question,
                    "correct_answer": self.get_final_answer(self.generate_COT_answer(question, chunk)),
                    "incorrect_answers": self.generate_bad_answers(question, chunk, num_answers)
                }
                dataset.append(data)
            if stop_early:
                break

        os.makedirs(os.path.dirname(self.output), exist_ok=True)
        print("copying to: ", self.output)
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        print("Done!")
        return dataset
