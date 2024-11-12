"""
This file will load pages from a Confluence Instance into a vector store along w/ their metadata
"""

from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from constants import *
from dotenv import load_dotenv
load_dotenv()
import os

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


class ConfluenceRAG:
    def __init__(self, pinecone_index: str, space_key: str = None, debug: bool = False):
        self.pinecone_index = pinecone_index
        self.space_key = space_key
        self.debug = debug

    def seed(self):
        """
        This will load all documents from a given space in Confluence and load them into the vector store
        """

        # Use ConfluenceLoader to load documents
        # Docs: https://python.langchain.com/v0.1/docs/integrations/document_loaders/confluence/
        loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            api_key=os.getenv("CONFLUENCE_API_TOKEN"),
            space_key=self.space_key,
            keep_markdown_format=True,
            keep_newlines=False,
        )
        raw_docs = loader.load()
        print(f"Loaded {len(raw_docs)} initial Documents (to be split)")

        self.split_and_load_docs(raw_docs)


    def upload_page(self, page_id: int):
        """
        Uploads (splits, embeds) a specific page.
        This can be used if we add or update a page and don't want to re-run for the entire space
        :param page_id:
        :return:
        """
        return self.upload_pages([page_id])


    def upload_pages(self, page_ids: list[int]):
        """
        Uploads (splits, embeds) a list of pages.
        :param page_ids:
        :return:
        """
        print(f"Loading pages {page_ids} into pinecone index {self.pinecone_index}...")
        loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            api_key=os.getenv("CONFLUENCE_API_TOKEN"),
            keep_markdown_format=True,
            keep_newlines=False,
            page_ids=page_ids
        )

        raw_docs = loader.load()
        print(f"Loaded {len(raw_docs)} Raw Documents (to be split)")
        self.split_and_load_docs(raw_docs)


    def load_documents(self, documents: Iterable[Document], namespace: str = None) -> None:
        print(f"Loading {len(documents)} documents into pinecone index {self.pinecone_index}...")
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=self.pinecone_index,
            namespace=namespace
        )
        print("DONE")


    def split_and_load_docs(self, documents: Iterable[Document]):
        # MarkdownTextSplitter
        # Note: this subclasses RecursiveCharacterTextSplitter, so it's quite similar
        # Docs: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/
        md_text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        md_text_splitter_documents = md_text_splitter.split_documents(documents)

        # Add additional metadata to the documents
        for doc in md_text_splitter_documents:
            doc.metadata["source_name"] = "confluence"
            doc.metadata["space"] = self.space_key

        # Load documents into Pinecone
        self.load_documents(documents=md_text_splitter_documents)


# Clearing documents, updating, etc (experimental)

def clear_documents_for_source(source: str) -> None:
    """
    This will delete all documents where the metadata[source] == source. This is useful if you want to reload a source.
    """
    print(f"Clearing documents for source {source}...")
    metadata_filter = {"source": source}

    pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pinecone.Index(PINECONE_INDEX)

    index_stats = index.describe_index_stats(filter=metadata_filter)

    # Extract document IDs from the index stats response
    document_ids = [id for id in index_stats['namespaces'][metadata_filter['namespace']]['ids']]

    print(f"Deleting the following document IDs: {document_ids}")
    vector_store.delete(ids=document_ids, index_name=PINECONE_INDEX)
    print("DONE")


def reload_document(doc: Document) -> None:
    # clear existing documents for the source data
    source = doc.metadata["source"]
    clear_documents_for_source(source)

    # split and load the document fresh
    split_and_load_docs([doc])


def reload_pages(page_ids: list[int]) -> None:
    """"
    This will remove existing documents for a given source,
    use a document Loader to fetch the documents and split again, and then
    import back into the vector store
    """
    loader = ConfluenceLoader(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        api_key=os.getenv("CONFLUENCE_API_TOKEN"),
        keep_markdown_format=True,
        keep_newlines=False,
        page_ids=page_ids
    )
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} initial Documents (to be split)")

    for doc in raw_docs:
        reload_document(doc)


def reload_page(page_id: int) -> None:
    print(f"Reloading page {page_id}")
    return reload_pages([page_id])
