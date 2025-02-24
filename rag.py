from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


set_debug(True)
set_verbose(True)


class Raggy:
    # model = ChatOllama
    # text_splitter = RecursiveCharacterTextSplitter
    # prompt = ChatPromptTemplate
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "llama3.2:1b"):
        self.model = ChatOllama(model=llm_model, temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but not always would it be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries but do not hallucinate.
        The context may be multiple different documents that are unrelated so do not try to form relations between them
        unless specifically asked.\n
        {context}""",
                ),
                (
                    "human",
                    "Here is the document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, fp: str, ft: str):
        docs = []

        if ft == "application/pdf":
            docs.extend(PyPDFLoader(file_path=fp).load())
        elif "officedocument" in ft:
            docs.extend(Docx2txtLoader(file_path=fp).load())
        elif ft == "text/plain":
            docs.extend(TextLoader(file_path=fp).load())
        elif ft == "application/octet-stream":
            docs.extend(TextLoader(file_path=fp).load())

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if self.vector_store:
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=FastEmbedEmbeddings(),
                persist_directory="chroma_db",
            )

    def ingest_url(self, url: str):
        docs = WebBaseLoader(url).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if self.vector_store:
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=FastEmbedEmbeddings(),
                persist_directory="chroma_db",
            )

    def ask(self, q: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding_function=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(q)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "Please add some documents"

        return self.chain.invoke(q)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
