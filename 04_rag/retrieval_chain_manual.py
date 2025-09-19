import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Retrieving...")

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    query = "What is pinecone in machine learning?"
    embeddings = OpenAIEmbeddings()

    llm = ChatOpenAI(model="gpt-4o-mini")

    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    # Chain creation to pass the formatted documents to the model. Basically fills in the context and returns chain.
    combined_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # Chain that executes retrieval of relevant documents, then execute the complete chain upon invocation
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combined_documents_chain,
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result)
