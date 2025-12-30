from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

PKG_ROOT = Path(__file__).resolve().parent.parent


if __name__ == '__main__':
    print("Faiss vector store!")
    pdf_path = PKG_ROOT / "04_rag" / "react.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)

    vector_store.save_local("faiss_index_react")

    new_vector_store = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    combined_documents = create_stuff_documents_chain(llm=OpenAI(), prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=new_vector_store.as_retriever(), combine_docs_chain=combined_documents)

    result = retrieval_chain.invoke(input={"input": "Give me the gist of ReAct in 3 sentences."})

    print(result["answer"])
