# %%
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from collections import defaultdict
import random
import json
import re

# %% [markdown]
# # Helper function

# %%
def get_chunk(path = "data", type = "md", chunk_size = 1500, chunk_overlap = 100):

    if type == "md":
        # Load all file ends with .md
        loader = DirectoryLoader(path, glob="**/[!.]*.md", loader_cls=UnstructuredMarkdownLoader)
    elif type == "pdf":
        loader = PyPDFDirectoryLoader(path)
    else:
        raise TypeError("Only accept pdf and md")
    
    chunks = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap))
    chunks = [(chunk.page_content, chunk.metadata['source']) for chunk in chunks]

    return chunks


def generate_questions(chunk, num = 3, model = "llama3"):
    """
    Generates `num` questions / use cases for `chunk`. Used when the input document is of general types 
    """
    messages=[
                {"role": "system", "content": "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and that question could be able to answer using information from the chunk. For example, if the given context has information about supercomputer, an example question could be 'What is a supercomputer?'" % (num)},
                {"role": "system", "content": "The questions should be able to be answered in a few words or less. Show the example questions in numbered list. Every questions MUST end with a question mark"},
                {"role": "user", "content": str(chunk)}
            ]

    response = client.chat.completions.create(
    model=model,
    messages=messages
    )
    queries = response.choices[0].message.content.split('\n')

    # Only include questions
    queries = [q for q in queries if q.endswith("?") and not (q.startswith("You are a synthetic"))]

    return [re.sub(r'^[\d+\.|*+\.]+\s', '', q) for q in queries] # If questions start with numbers or stars, remove them.
    

def encode_question(question, chunk):
    """
    Encode multiple prompt instructions into a single string.
    """
    
    prompts = []
        
    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=str(chunk))
    prompts.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    prompts.append({"role": "user", "content": prompt})
    return prompts

def generate_label(question, chunk, model = "llama3"):
    """
    Generates the label / answer to `question` using `context`.
    """
    question = encode_question(question, chunk)
    response = client.chat.completions.create(
        model=model,
        messages=question,
        n=1,
        temperature=0
    )
    response = response.choices[0].message.content
    return response

def get_final_answer(queries):
    beg = "<ANSWER>:"
    try:
        start = queries.rindex(beg)
        queries = queries[start+len(beg)+1:]
    except:
        pass

    return queries


def run(i, chunks, chunk, source, num = 3, num_distract = 4, p = 0.8, model = "llama3"):
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    res = []
    qs = generate_questions(chunk, num, model)
    for j, q in enumerate(qs):
        datapt = {
            "id": None,
            "context": None,
            "golden_context": None,
            "question": None,
            "cot_answer": None,
            "answer" : None
        }

        datapt["id"] = f"{source}_seed_task_{i}_{j}"
        datapt["question"] = q

        # add num_distract distractor docs
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for k in random.sample(indices, num_distract):
            docs.append(chunks[k][0])
            
        # decides whether to keep golden document
        golden = random.uniform(0, 1) < p
        if not golden:
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        datapt["context"] = docs
        datapt["golden_context"] = chunk

        # add answer to q
        cot_answer = generate_label(q, chunk, model=model) 
        datapt["cot_answer"] = cot_answer
        datapt["answer"] = get_final_answer(cot_answer)

        res.append(datapt)
    return res        

# %%
CHUNK_SIZE = 2000
NUM_DISTRACT_DOCS = 3
CHUNK_OVERLAP = 100
NUM_QUESTION = 3
P = 0.8 # chance of including golden document in training set
MODEL = "llama3-chatqa:70b"

# %%
# init OpenAI client
client = OpenAI(
    base_url = 'http://localhost:11434/v1', # remove this line if using gpt
    api_key='ollama', # [ollama, OPENAI_API_KEY] local LLM or using gpt
)

# %%
data = []
path = "data/md/polaris"
type = "md"
chunks = get_chunk(path, type, CHUNK_SIZE, CHUNK_OVERLAP)

# generate questions, answer and 4 distract documents
for i, (chunk, source) in enumerate(chunks):
    data.extend(run(i=i, chunks=chunks, chunk=chunk, source=source, num_distract=NUM_DISTRACT_DOCS))

# %%
# save data to a json file
out_path = f'output/QA_polaris_md_{MODEL}_{CHUNK_SIZE}_{NUM_DISTRACT_DOCS}.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)