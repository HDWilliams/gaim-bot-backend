from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader
from pinecone import Pinecone
from os import getenv
from dotenv import load_dotenv
from typing import Optional
from uuid import uuid4

load_dotenv()
openai_api_key = getenv('OPENAI_API_KEY')

def ingest_document(filename: str, chunk_size=200, chunk_overlap=50, index_name:str = 'gaim-bot-index') -> PineconeVectorStore:
    """ ingest a .csv or .txt file and return a FAISS index, or modify existing index """
    _, ext = filename.split('.') 

    if ext == 'csv':
        loader = CSVLoader(filename)
    elif ext == 'txt':
        loader = TextLoader(filename)
    else:
        raise TypeError('incorrect file type')

    documents = loader.load()

    # Split documents for embedding


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(documents)

    # Create FAISS index
    pc = Pinecone(api_key=getenv('PINECONE_API_KEY'))
    pc_index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)

    embeddings = OpenAIEmbeddings()
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=split_documents, ids=uuids)

    return vector_store


if __name__ == '__main__':
    ingest_document('locations.csv')



