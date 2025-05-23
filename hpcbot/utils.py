from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredRSTLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
import random
import re

def get_chunks(document_dir, chunk_size=1500, chunk_overlap=100):
    # Ensure chunk_size and chunk_overlap are integers
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    # Load documents from the input directory
    txt_loader = DirectoryLoader(document_dir, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(document_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    md_loader = DirectoryLoader(document_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    rst_loader = DirectoryLoader(document_dir, glob="**/*.rst", loader_cls=UnstructuredRSTLoader)

    if txt_loader or pdf_loader or md_loader or rst_loader:
        # Load and combine documents
        loader = txt_loader.load() + pdf_loader.load() + md_loader.load() + rst_loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(loader)
        print("Load success")

        return [(chunk.page_content, chunk.metadata['source']) for chunk in chunks]

    else:
        raise TypeError("Only 'txt', 'rst', 'md' and 'pdf' are supported.")
    

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