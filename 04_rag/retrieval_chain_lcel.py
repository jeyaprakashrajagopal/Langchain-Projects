import os
from typing import List

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_documents(documents: List[Document]):
    """Formats list of documents of type Document to a string representation"""
    return "\n\n".join([document.page_content for document in documents])


if __name__ == "__main__":
    print("Retrieving...")

    # Create instances of llm and embeddings
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings()

    prompt = """
    Please answer the question solely based on the context given below.
    Return you don't the answer if the question is not answerable with the context given below and do not generate any answers by yourself.
    Always say thanks for asking after the answer is shown.

    {context}

    user input: {question}

    Helpful answer: 
    """

    query = "What is pinecone in machine learning?"
    # Load PromptTemplate
    prompt = PromptTemplate.from_template(template=prompt)

    # PineCone vector store instance
    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # Chain the actions together where output of retriever with relevant documents are formatted
    chain = (
        {
            "context": vector_store.as_retriever() | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    result = chain.invoke(query)

    print(result.content)
