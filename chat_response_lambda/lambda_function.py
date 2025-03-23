"""
Main function of AWS Lambda function. Note: Need to import all packages into this function
"""
import time
import json
import logging
from os import getenv, path, makedirs, getcwd

from typing import List

from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone

import openai

from dotenv import load_dotenv

from utilities.RateLimiter import RateLimiter

from retry_with_exp_backoff import retry_openai_with_exp_backoff
from open_ai_interfaces import make_openai_request, get_openai_response
import PineconeInterface

def lambda_handler(event, context):
    """Manages lambda event. returns dict as json object

    Args:
        event: obj
        content: obj

    Returns:
        {'statusCode: int, 'body': json}
    """
    try:
        event_json = json.loads(event['body'])
        message_history = event_json['messages']
        index_name = event_json['index_name']
        if not message_history:
            raise ValueError('No message was included with request')
        output = get_openai_response(message_history, index_name)


        return {
            'statusCode': 200,
            'body': json.dumps(output)
        }
    # TODO: catch more specific exceptions
    except Exception as error:
        return {
            'statusCode': 400,
            'body': json.dumps({'completed': False, 'data': str(error)})
        }

