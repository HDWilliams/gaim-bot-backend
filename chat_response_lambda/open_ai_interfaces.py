import logging
from os import getenv
from typing import List
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone

import openai

from ratelimit import limits

from PineconeInterface import PineconeInterface
from retry_with_exp_backoff import retry_openai_with_exp_backoff

@limits(calls=2, period=1)
@retry_openai_with_exp_backoff
def make_openai_request(history:List[dict], model='gpt-4o-mini') -> str:
    load_dotenv()
    
    openai.api_key = getenv('OPENAI_API_KEY')
    client = openai.Client()
    response = client.chat.completions.create(
        model=model,
        messages=history
    )
    return response.choices[0].message.content

def add_retrieved_to_message_history(messages: List[dict], retrieved_docs: str) -> List[dict]:
    history = messages
    history[-1]['content'] += ' Context information: ' + retrieved_docs
    return history

    

def get_openai_response(messages: List[dict], index_name:str, model='gpt-4o-mini') -> dict:
    load_dotenv()

    pinecone_interface = PineconeInterface(index_name)
    
    retrieved_docs = pinecone_interface.retrieve_similar(messages)
    if retrieved_docs is None:
        return {'completed': False, 'data': 'A pinecone error occured'}
    message_history = add_retrieved_to_message_history(messages, retrieved_docs)

    return make_openai_request(message_history, model=model) 


