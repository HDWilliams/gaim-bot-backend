import time
import logging
from os import getenv, path, makedirs, getcwd

from typing import List

from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone

import openai
from openai._exceptions import APIConnectionError, RateLimitError, APIError, APIStatusError, BadRequestError
from requests.exceptions import RequestException

from dotenv import load_dotenv

from RateLimiter import RateLimiter

from retry_with_exp_backoff import retry_with_exp_backoff


"""
1. load faiss index
2. get documents from query
3. send query and context to api
4. return response
"""

def get_pinecone_vector_store(index_name:str='gaim-bot-index') -> PineconeVectorStore:

    load_dotenv()

    pc_api_key = getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pc_api_key)
    pc_index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)

    return vector_store


@retry_with_exp_backoff
def make_openai_request(prompt, model='gpt-4o-mini') -> str:
    load_dotenv()

    openai.api_key = getenv('OPENAI_API_KEY')
    client = openai.Client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content':prompt}]
        )
        return response.choices[0].message.content

    except RateLimitError as err_rl:
        logging.error(f'Rate limit error {err_rl}')

        remaining_requests = int(err_rl.response.headers.get('x-ratelimit-remaining-requests', 10000))
        remaining_tokens = int(err_rl.response.headers.get('x-ratelimit-remaining-tokens', 200000))

        if remaining_requests == 0 and remaining_tokens == 0:
            retry_after = int(err_rl.response.headers.get('Retry-After', 10))
            time.sleep(retry_after)

    except APIConnectionError as err_connection:
        logging.error('API connection error: %s', err_connection)
    except APIStatusError as err_status:
        logging.error(f'API Status Error {err_status}')
        # Get headers from OpenAI api for token/rate limit reset
    except APIError as err_api:
        err_status.response.headers
        logging.error(f'API Error {err_api}')
    except RequestException as err_ex:
        logging.error(f'Error occurred {err_ex}')
    

def get_openai_response(query: str, index_name:str='gaim-bot-index', model='gpt-4o-mini', rate_limit=500, period=1, token_limit=30000) -> dict:
    
    prompt = ChatPromptTemplate([('system', 'You are a helpful videogame guide for the videogame Eldin Ring. Please answer questions related to the game using your knowledge and the included context.'), ('human', 'question: {query}, context: {context}')])
    
    vector_store = get_pinecone_vector_store(index_name)
    context = vector_store.similarity_search(query, k=2)
    return make_openai_request(prompt.format(query=query, context=context), model=model)   


if __name__ == "__main__":
    print(get_openai_response('whats eldin ring'))