

from dotenv import load_dotenv
load_dotenv()

from constants import *

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

class AskDocumentation:
    def __init__(self, pinecone_index: str):
        self.pinecone_index = pinecone_index

        self._llm = ChatOpenAI(temperature=0.7)
        self._embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._vector_store = PineconeVectorStore(index_name=self.pinecone_index, embedding=self._embeddings)
        self._retriever = self._vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.6},
            k=8
        )

    def ask(self, query: str) -> str:
        """
        Ask a question and get a formatted response from the LLM
        :param query:
        :return:
        """

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
                {
                    "context": itemgetter("question") | self._retriever, "question": itemgetter("question")
                }
                | prompt
                | self._llm
                | StrOutputParser()
        )

        return chain.invoke({"question": query})



def main():
    asker = AskDocumentation(pinecone_index=PINECONE_INDEX)

    while True:
        q = input("Enter your query (or type 'q' to quit): ")
        if q.lower() == 'q':
            break

        print(f"Q: {q}")
        rsp = asker.ask(q)
        print(f"A: {rsp}\n\n")

if __name__ == '__main__':
    main()