Project Notes
=============

My plan is to create a Q+A AI tool that uses our internal knowledge.

### My "Minimum Requirements" version will be:
- Load, Spit, Embed our Confluence Documentation ("Engineering" Space and "PM" Space)
  - https://python.langchain.com/v0.1/docs/integrations/document_loaders/confluence/
- Load our "Kimkim Academy" (data from a Postgres DB)
- Use RAG to answer questions from the CLI

### My Stretch Goals
- Add additional datasources and tag w/ metadata
  [X] PM space
  [X] Internal "Academy Content"
  [ ] Google Docs (via list for now) just to try it
  [ ] Rails Code
  [ ]Documentation of common libraries
  [ ]Jira (via list for now)
- Auto refresh content as it changes or new pages added to confluence (vs delete and rebuild)

* * * 

## Running The Code

### Seeding Vector Store
We will load/split/chunk Confluence into a Pinecone Vector Store for this project.

```
# Run this to seed the 2 Confluence Spaces and Academy Content stored in a Postgres DB
./upload_sources.py

def seed_all(index):
    print("Seeding all spaces...")

    # Seed the Engineering Space
    ConfluenceRAG(pinecone_index=index, space_key="ENGINEERIN").seed()

    # Seed the Product Space
    ConfluenceRAG(pinecone_index=index, space_key="PM").seed()

    # Load the Academy Articles
    AcademyRAG(pinecone_index=index).seed()

    print("DONE")
```

### Querying via CLI
Run `rag_qa.py`, this will prompt you for a question.

This uses `AskDocumentation` to connect w/ Pinecone and send the documents to the LLM

### Testing
A few helpers / test questions:
- `test_questions.py`
- `test_retrieve.py`


* * * 

## Code Structure / Helpers
A few helper classes to wrap the logic.

### `ConfluenceRAG`
Uses `ConfluenceLoader` to load raw pages from a Confluence account (and space), 
then uses the `MarkdownTextSplitter` to split and `PineconeVectorStore` to load.

We use this for both the "Engineering" space and "Product/PM" space to load everything to the same Pinecone index 
(w/ metadata to separate/query/filter as needed).


### `AcademyRAG`
Uses `PostgresLoader` to load a Postgres SQL query w/ metadata into documents, then loads via `PineconeVectorStore`


### `AskDocumentation`
Retrieves relevant documents for a query from Pinecone, sends the context to the LLM to answer the question

### `Retriever`
A simple class to query the Pinecone index and return the top results. 
(Used a bit for debugging the results, scores, etc)

* * * 

## Embedding Metadata
Store info that I can use to filter on or display later
- Original source, updated timestamp, tags


### Additional Reading
- https://www.shakudo.io/blog/building-confluence-kb-qanda-app-langchain-chatgpt

 * * * 

# Improving
## RAG
- Look at a few documents uploaded and split to see if it makes sense (Confluence, and code)
- Try to get markdown splitter working better w/ code blocks and # within them.
- Self Query? https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/


 * * * 

## Steps
### Connect to Confluence Instance
- [x] Access
- [x] Print High Level stats ("Loaded 102 files")

#### Notes
- Needed to `pip install lxml`
- Example docs were outdated, moved params to initializer (not in `.load`)

### MarkdownHeaderTextSplitter
You need to `pip install markdownify` and set `keep_markdown_format=True` otherwise all # and other formatting will be strippped as part of the loading process (and you need that for the Markdown Text splitter to work)

### Embed Documents, Import in VectorStore
- [] Local FAISS for testing purposes (does it really matter if I need to use OpenAI to embed?)
- [] Different namespaces for chunking techniques (Read more about this and querying)
- [] Pinecone for production


### Stats
Loaded 102 files
RecursiveCharacterTextSplitter: Going to add 505 documents to the vector store
MarkdownTextSplitter: Going to add 505 documents to the vector store
MarkdownHeaderTextSplitter: Going to add 626 documents to the vector store

Loading 505 documents into namespace RecursiveCharacterTextSplitter...
Loading 505 documents into namespace MarkdownTextSplitter...
Loading 626 documents into namespace MarkdownHeaderTextSplitter...


## TODOs/Followups
- I didn't see any of the embedding / importing into Pinecone in LangSmith.
- Even though I used these langchain packages
- from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

and had my tracing + key set in .env

- If you supply `page_ids`, it will not use it if `space` is also set.