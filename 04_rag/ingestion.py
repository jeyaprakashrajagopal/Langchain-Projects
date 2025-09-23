import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()

if __name__ == "__main__":
    print("RAG")
    # Step 1 Ingestion : load the text from the file
    loader = TextLoader(
        PKG_ROOT / "04_rag" / "medium_blog.txt"
    )
    document = loader.load()

    # Step 2: split document into chunks without any context shared between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)

    # Step 3 splitting : create embeddings using the default model text-embedding-ada-002
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Step 4 ingesting : generate and store embeddings
    PineconeVectorStore.from_documents(
        documents=texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
