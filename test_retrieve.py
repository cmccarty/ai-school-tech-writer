from retrieve import Retriever
from test_questions import *

# Test the document retrieval (w/ scores to help w/ debugging)
retriever = Retriever(debug=True)
docs = retriever.retrieve_relevant_docs_with_scores(TEST_QUESTION, k=10)

# Print the documents and scores
print("\n\n\n")
print(f"Found {len(docs)} documents for query: {TEST_QUESTION}")
for doc, score in docs:
    print(f"Score: {score}\nMetadata: {doc.metadata}\nContent: {doc.page_content}\n---------\n\n")
