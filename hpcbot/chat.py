# hpcbot/chat.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredRSTLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

class ChatBot:
    def __init__(self, model, document_dir, api_key, base_url, persist_directory = "./chromaDB"):
        self.model = model
        self.document_dir = document_dir
        self.persist_directory = persist_directory
        self.retriever = self._setup_retriever()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _setup_retriever(self):
        if os.path.exists(self.persist_directory):
            # Load the existing database
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
            )
        else:
            # Load documents from the directory (include .txt, .pdf, .md, and .rst files)
            txt_loader = DirectoryLoader(self.document_dir, glob="**/*.txt", loader_cls=TextLoader)
            pdf_loader = DirectoryLoader(self.document_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            md_loader = DirectoryLoader(self.document_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
            rst_loader = DirectoryLoader(self.document_dir, glob="**/*.rst", loader_cls=UnstructuredRSTLoader)

            # Combine all documents
            loader = txt_loader.load() + pdf_loader.load() + md_loader.load() + rst_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            documents = text_splitter.split_documents(loader)

            # Create embeddings and vector store
            db = Chroma.from_documents(
                documents, 
                OllamaEmbeddings(model="mxbai-embed-large", show_progress=True),
                persist_directory=self.persist_directory
                )
            db.persist()
        
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask_openai(self, input_dict):
        """
        Call the OpenAI API with the given input dictionary.
        
        Args:
            input_dict (dict): A dictionary containing "prompt" (the formatted prompt).
        
        Returns:
            str: The generated response from the OpenAI API.
        """
        question = input_dict["question"]
        context = input_dict["context"]

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "system","content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise."},
                {"role": "user","content": f"Question: {question}\nContext: {context}"}
            ],
            temperature = 0.2
        )
        return response.choices[0].message.content

    def ask(self, question: str) -> str:
        
        # Set up the QA chain
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(self.ask_openai)
            | StrOutputParser()
        )
        # Run the chain and return the answer
        return rag_chain.invoke(question)