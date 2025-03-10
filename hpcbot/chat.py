# hpcbot/chat.py
import os
import logging
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredRSTLoader,
)
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(
        self, model, document_dir, api_key, base_url, persist_directory="./chromaDB"
    ):
        logger.debug(
            "Initializing ChatBot with model=%s, document_dir=%s", model, document_dir
        )
        self.model = model
        self.document_dir = document_dir
        self.persist_directory = persist_directory
        self.retriever = self._setup_retriever()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.debug("ChatBot initialization complete")

    def _setup_retriever(self):
        logger.debug(
            "Setting up retriever with persist_directory=%s", self.persist_directory
        )
        if os.path.exists(self.persist_directory):
            # Load the existing database
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=OllamaEmbeddings(
                    model="mxbai-embed-large", show_progress=True
                ),
            )
        else:
            # Load documents from the directory (include .txt, .pdf, .md, and .rst files)
            txt_loader = DirectoryLoader(
                self.document_dir, glob="**/*.txt", loader_cls=TextLoader
            )
            pdf_loader = DirectoryLoader(
                self.document_dir, glob="**/*.pdf", loader_cls=PyPDFLoader
            )
            md_loader = DirectoryLoader(
                self.document_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
            )
            rst_loader = DirectoryLoader(
                self.document_dir, glob="**/*.rst", loader_cls=UnstructuredRSTLoader
            )

            loaders = [txt_loader, pdf_loader, md_loader, rst_loader]

            documents = []
            for loader in loaders:
                docs = loader.load()  # Load each file as a full document
                documents.extend(docs) 
            """    
            # Combine all documents
            loader = (
                txt_loader.load()
                + pdf_loader.load()
                + md_loader.load()
                + rst_loader.load()
            )
            # Split
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=100
            )
            documents = text_splitter.split_documents(loader)
            """
            # Not-spliting
             

            # Create embeddings and vector store
            db = Chroma.from_documents(
                documents,
                OllamaEmbeddings(model="mxbai-embed-large", show_progress=True),
                persist_directory=self.persist_directory,
            )
            db.persist()

        return db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

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
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nContext: {context}",
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def ask(self, question: str) -> str:
        logger.debug("Processing question: %s", question)

        # Get retriever results
        retriever_results = self.retriever.invoke(question)
        logger.debug("Retrieved %d documents", len(retriever_results))
        for i, doc in enumerate(retriever_results):
            logger.debug("Document %d: %s", i + 1, doc.page_content[:200] + "...")

        # Set up the QA chain
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.ask_openai)
            | StrOutputParser()
        )

        # Get and log the answer
        answer = rag_chain.invoke(question)
        logger.debug("Generated answer: %s", answer)
        return answer
