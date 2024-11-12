"""
This file will connect to Pinecone to pull the relevant documents for a given query
And then feed this context into an LLM to better compile a full answer.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Load ENV variables required by the libraries
from dotenv import load_dotenv
load_dotenv()

from constants import *


# Setup the LLM and Embeddings model
llm = ChatOpenAI(temperature=0.7)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


class Retriever:
    def __init__(self, debug: bool = False):
        self.debug = debug


    def retrieve_relevant_docs(self, query: str) -> list[Document]:
        """
        Query the Pinecone index for relevant documents
        :param query: The query to search for
        :return: A list of documents
        """
        if self.debug:
            print(f"Searching for relevant documents for query:\n\t{query}")
        document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
        retriever = document_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.6},
            k=8
        )
        context = retriever.invoke(query)
        return context


    def retrieve_relevant_docs_with_scores(self, query: str, k: int = 4) -> list[tuple]:
        """
        Query the Pinecone index for relevant documents
        :param query: The query to search for
        :return: A list of tuples with the document and its score
        """
        if self.debug:
            print(f"Searching for relevant documents for query:\n\t{query}")

        document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
        context = document_vectorstore.similarity_search_with_relevance_scores(query=query, k=k)
        return context


def perform_rag_llm_prompt(query: str, pinecone_namespace: str = None) -> str:
    # get the context
    context = retrieve_relevant_docs(query, pinecone_namespace)

    # Print context w/ scores, source info, and metadata
    for doc in context:
        # print(doc)
        print(f"Metadata: {doc.metadata}\nContent: {doc.page_content}\n\n")

    # Adding context to our prompt
    template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
    prompt_with_context = template.invoke({"query": query, "context": context})

    # Asking the LLM for a response from our prompt with the provided context
    results = llm.invoke(prompt_with_context)
    return results.content
