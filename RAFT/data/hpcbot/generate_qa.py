from .utils import get_chunks, generate_questions, generate_COT_answer, generate_bad_answers, get_final_answer
import json
import random
import yaml


from openai import OpenAI

class Config:
    def __init__(self):
        self.config = {}

    def load_config(self, config_file):
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def get_config(self, key):
        return self.config.get(key)


config = Config()
config.load_config('/lus/eagle/projects/HPCBot/workspace/package/hpcbot/config.yaml')

is_local = config.get_config('is_local')
ACCESS_TOKEN = config.get_config('access_token')
BASE_URL = config.get_config('base_url')
MODEL = config.get_config('model')

if is_local:
    ACCESS_TOKEN = 'ollama'
    BASE_URL = 'http://localhost:11434/v1'
else:
    if not ACCESS_TOKEN:
        with open('/lus/eagle/projects/HPCBot/workspace/package/hpcbot/access_token.txt', 'r') as file:
            ACCESS_TOKEN = file.read().strip()
    

class QAContextDistractors:
    def __init__(self, model=MODEL, api_key=ACCESS_TOKEN, base_url=BASE_URL):
        '''
        if using openAI, set base_url = None, api_key = OPENAI_API_KEY
        '''
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def get_chunks(self, path, file_type="md", chunk_size=1500, chunk_overlap=100):
        return get_chunks(path, file_type, chunk_size, chunk_overlap)

    def generate_questions(self, chunk, num=3):
        return generate_questions(self.client, self.model, chunk, num)
    
    def generate_COT_answer(self, question, chunk):
        return generate_COT_answer(self.client, self.model, question, chunk)

    def get_final_answer(self, queries):
        return get_final_answer(queries)
    
    def run(self, path = "data", file_type = "md", output = "output/QA_context_distractor.json", num_questions=3, num_distractors=3, include_gold=0.8, stop_early = False):
        dataset = []
        print("getting chunks")
        chunks = self.get_chunks(path, file_type)
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
        print("copying to: ", output)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        print("Done!")
        return dataset
        
class QAAnswerDistractors:
    def __init__(self, model=MODEL, api_key=ACCESS_TOKEN, base_url=BASE_URL):
        '''
        if using openAI, set base_url = None, api_key = OPENAI_API_KEY
        '''
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def get_chunks(self, path, file_type, chunk_size=1500, chunk_overlap=100):
        return get_chunks(path, file_type, chunk_size, chunk_overlap)

    def generate_questions(self, chunk, num=3):
        return generate_questions(self.client, self.model, chunk, num)
    
    def generate_COT_answer(self, question, chunk):
        return generate_COT_answer(self.client, self.model, question, chunk)

    def get_final_answer(self, queries):
        return get_final_answer(queries)
    
    def generate_bad_answers(self, question, chunk, num_answer):
        return generate_bad_answers(self.client, self.model, question, chunk, num_answer)
    
    def run(self, path = "data", file_type = "md", output = "output/QA_answer_distractor.json", num_questions = 3, num_answers = 4, stop_early = False):
        dataset = []
        print("getting chunks")
        chunks = self.get_chunks(path, file_type)
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

        print("copying to: ", output)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        print("Done!")
        return dataset
