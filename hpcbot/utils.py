from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
import random
import re

def get_chunks(path, file_type="md", chunk_size=1500, chunk_overlap=100):
    if file_type == "md":
        loader = DirectoryLoader(path, glob="**/[!.]*.md", loader_cls=UnstructuredMarkdownLoader)
    elif file_type == "pdf":
        loader = PyPDFDirectoryLoader(path)
    else:
        raise TypeError("Only 'md' and 'pdf' are supported.")

    chunks = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return [(chunk.page_content, chunk.metadata['source']) for chunk in chunks]

def generate_questions(client, model, chunk, num=3):
    '''
    Generates `num` questions / use cases for `chunk`. Used when the input document is of general types 
    '''
    messages=[
            {"role": "system", "content": "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and that question could be able to answer using information from the chunk. For example, if the given context has information about supercomputer, an example question could be 'What is a supercomputer?'" % (num)},
            {"role": "system", "content": "The questions should be able to be answered in a few words or less. Show the example questions in numbered list. Every questions MUST end with a question mark"},
            {"role": "user", "content": str(chunk)}
        ]
    response = client.chat.completions.create(model=model, messages=messages)
    return [re.sub(r'^\d+\.\s', '', q) for q in response.choices[0].message.content.split('\n') if q.endswith("?")]

def generate_COT_answer(client, model, question, chunk):
    '''
    generate chain of thought correct answers
    '''
    prompts = []
    
    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=str(chunk))
    prompts.append({"role": "system", "content": "You are a helpful assistant answering questions using provided context."})
    prompts.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(model=model, messages=prompts, temperature=0.1)
    return response.choices[0].message.content

def generate_bad_answers(client, model, question, chunk, num_answer = 4):
    '''
    generate {num_answer} bad answers
    '''
    prompts = []
        
    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question incorrectly in {num_answer} ways. 
        The incorrect answers should be succinct.        
    """.format(question=question, context=str(chunk), num_answer=str(num_answer))
    prompts.append({"role": "system", "content": "You are a not helpful assistant answering questions wrong using provided context"})
    prompts.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=prompts,
        temperature=0.5 # increase temperature for LLM be more creative with incorrect answer
    )

    queries = response.choices[0].message.content.split('\n')
    pattern = r'^[\d+\.|\*+\.|\*\*Answer:\*\*|\d+\.+\s+\*\*Answer:\*\*]+\s'
    
    return [re.sub(pattern, '', a) for a in filter(None, queries) if a[0].isdigit()]    

def get_final_answer(queries):
        beg = "<ANSWER>:"
        try:
            start = queries.rindex(beg)
            queries = queries[start+len(beg)+1:]
        except:
            pass

        return queries