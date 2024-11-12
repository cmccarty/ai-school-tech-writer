from typing import Iterable
from langchain_core.documents import Document
from postgres_loader import PostgresLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from constants import *
from dotenv import load_dotenv
load_dotenv()

class AcademyRAG:
    """
    Loader for the Kimkim Academy articles, loads into Pinecone Vector Store
    """

    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        pass

    def seed(self):
        """
        Loads local postgres DB w/ query to load our "Academy" Articles,
        which are How Tos documentation for how to use certain aspects of the product
        """
        loader = PostgresLoader(db_url="postgres://kimkim:@localhost:5432/kimkim_dev")
        raw_docs = loader.load(
            query="""
                SELECT 
                    academy_contents.*
                    , academy_contents.updated_at AS when
                    , academy_content_categories.name as category 
                FROM 
                    academy_contents 
                LEFT JOIN 
                    academy_content_categories ON academy_content_categories.id = category_id 
                WHERE status = 1
            """,
            content_columns=['content'],
            metadata_columns=['id', 'slug', 'title', 'category', 'when']
        )

        print(f"Loaded {len(raw_docs)} initial Documents (to be split)")
        self._split_and_load_docs(raw_docs)


    def _load_documents(self, documents: Iterable[Document], namespace: str = None) -> None:
        print(f"Loading {len(documents)} documents into pinecone index {self.pinecone_index}...")

        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.pinecone_index,
            namespace=namespace
        )

        print("DONE")


    def _split_and_load_docs(self, documents: Iterable[Document]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_splitter_documents = text_splitter.split_documents(documents)

        print(f"Loaded {len(documents)} initial Documents, split into {len(text_splitter_documents)} chunks")


        # Add additional metadata to the documents
        for doc in text_splitter_documents:
            doc.metadata["source_name"] = "academy"
            if doc.metadata["slug"]:
                doc.metadata["source"] = f"https://www.kimkim.com/academy/{doc.metadata['slug']}"


        # Load documents into Pinecone
        self._load_documents(documents=text_splitter_documents)