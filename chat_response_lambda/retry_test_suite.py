import sys
import os


from typing import List
from pprint import pprint

from openai import APIError, APIStatusError, RateLimitError, APIConnectionError, BadRequestError
from requests.models import Request, Response
from retry_with_exp_backoff import retry_openai_with_exp_backoff
from open_ai_interfaces import get_openai_response
 



@retry_openai_with_exp_backoff
def generate_error(exeception: APIError) -> None:
    raise exeception

def test_all_exceptions(exception: Exception) -> bool:

    print(f'Testing OpenAI api method with {str(exception)}')
    output = generate_error(exception)

    print('\n')
    
    pprint(f'The function provided the following output: {output}')

    print('\n\n')


if __name__ == "__main__":
    retry_status_codes_to_test=[400, 500, 501, 502, 503]

    test_req = Request()
    test_res = Response()

    test_res.request = test_req
    
    test_res.headers['x-request-id'] = 1
    
    for status_code in retry_status_codes_to_test:

        error_message = f'A APIStatusError occured with status code: {status_code}'
        test_res.status_code = status_code

        api_status_error = APIStatusError(message=error_message, response=test_res, body=None)
        test_all_exceptions(api_status_error)
    
    error_message = f'A Ratelimit error occured with status code 429'
    ratelimit_error = RateLimitError(message=error_message, response=test_res, body=None)
    exepected_error_output = {'completed': False, 'data': str(ratelimit_error)}
    print(f'Expected output: {exepected_error_output}')
    test_all_exceptions(ratelimit_error)
    

    error_message = f'A bad request error occured'
    test_res.status_code = 400
    bad_request_error = BadRequestError(message=error_message, response=test_res, body=None)
    exepected_error_output = {'completed': False, 'data': str(bad_request_error)}
    print(f'Expected output: {exepected_error_output}')
    test_all_exceptions(bad_request_error)

    error_message = f'An api connection error occured'
    api_connection_error = APIConnectionError(request=test_req)
    exepected_error_output = {'completed': False, 'data': str(api_connection_error)}
    print(f'Expected output: {exepected_error_output}')
    test_all_exceptions(bad_request_error)

    # TEST REQUEST TO OPENAI
    msgs = [{
        'role': 'user',
        'content': 'Whats eldin ring'
    }]
    index = 'gaim-bot-index'
    print(get_openai_response(msgs, index))


    
    
    
