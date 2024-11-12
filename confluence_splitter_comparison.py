"""
This code will split the Confluence Documents using different splitter methods and upload them to the vector store
Then we can test which are better.

My initial test w/ the MarkdownHeader version were not successful, even though I think it has the most potential in theory since
we split by different topic/subtopic.

One issue was that the splitter wasn't smart enough to ignore a # within a ``` code block. which created some weird headers and hierarchies.

The retrieval stop also returned unexpected results.

Due to this, I switched to the general recursive splitter for my v1 testing.

"""

from langchain_core.documents import Document
from typing import Iterable, Any, Dict, List, Tuple, TypedDict

from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from constants import *
from dotenv import load_dotenv
load_dotenv()
import os

from mergedeep import merge

PINECONE_INDEX = "confluence-rag-splitters"



# Use ConfluenceLoader to load documents
# Docs: https://python.langchain.com/v0.1/docs/integrations/document_loaders/confluence/
loader = ConfluenceLoader(
    url=os.getenv("CONFLUENCE_URL"),
    username=os.getenv("CONFLUENCE_USERNAME"),
    api_key=os.getenv("CONFLUENCE_API_TOKEN"),
    space_key="ENGINEERIN",
    keep_markdown_format=True,
    keep_newlines=False,
    # max_pages=10
)

print("Loading documents...")
raw_docs = loader.load()

# Print some stats (for testing)
print(f"Loaded {len(raw_docs)} initial Documents (to be split)")

# Compare different splitting/chunking techniques
## RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_splitter_documents = text_splitter.split_documents(raw_docs)
print(f"RecursiveCharacterTextSplitter: Going to add {len(text_splitter_documents)} documents to the vector store")


## MarkdownHeaderTextSplitter
## Docs: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/
md_text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
md_text_splitter_documents = md_text_splitter.split_documents(raw_docs)
print(f"MarkdownTextSplitter: Going to add {len(md_text_splitter_documents)} documents to the vector store")

def split_markdown_documents(documents: Iterable[Document], splitter: MarkdownHeaderTextSplitter) -> List[Document]:
    #texts, metadatas = [], []

    split_docs = []
    for doc in documents:
        markdown_document = doc.page_content
        document_metadata = doc.metadata

        md_header_splits = markdown_splitter.split_text(markdown_document)

        # Merge document metadata w/ any metadata added from the splitting process
        for split in md_header_splits:
            merged_metadata = merge(document_metadata, split.metadata)
            split.metadata = merged_metadata

            # add split Document and metadatas to the full lists
            split_docs.append(split)

    return split_docs



# Split on specific headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
markdown_header_documents = split_markdown_documents(documents=raw_docs, splitter=markdown_splitter)
print(f"MarkdownHeaderTextSplitter: Going to add {len(markdown_header_documents)} documents to the vector store")

# Embed and upload to Pinecone
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# We will use 3 separate namespaces, 1 for each of the splitting/chunking methods so that we can compare
# These will all be in 1 index

def load_documents(documents: Iterable[Document], namespace: str) -> None:
    print(f"Loading {len(documents)} documents into namespace {namespace}...")
    PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name=PINECONE_INDEX, namespace=namespace)
    print("DONE")

# Load RecursiveCharacterTextSplitter documents
load_documents(documents=text_splitter_documents, namespace="RecursiveCharacterTextSplitter")

# Load MarkdownTextSplitter documents
load_documents(documents=md_text_splitter_documents, namespace="MarkdownTextSplitter")

# Load MarkdownHeaderTextSplitter documents
load_documents(documents=markdown_header_documents, namespace="MarkdownHeaderTextSplitter")

print("DONE")