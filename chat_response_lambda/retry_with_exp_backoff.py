from openai import RateLimitError, APIStatusError, APIConnectionError, APIError, APITimeoutError
import random
import time
from typing import Tuple, Callable
import logging

def calculate_delay(retry_num: int, delay: float, exp_base: float, jitter: bool) -> float:
    delay *= exp_base**(retry_num - 1)
    if jitter:
        delay += random.uniform(0,1)
    
    return delay

def retry_openai_with_exp_backoff(func: Callable, jitter:bool=True, 
                           init_delay:float=1, 
                           exp_base:int=1.3,
                           max_retries:int=3, 
                           retry_code_list = (500, 502, 503, 504),
                           openai_ratelimit_retry_time = 60,
                           ):
    def wrapper(*args, **kwargs):

        retry_num = 0
        delay = init_delay
        while retry_num <= max_retries:
            try:
                chat_output = func(*args, **kwargs)
                return {'completed': True, 'data': chat_output}
            except RateLimitError as err_rate:
                logging.error(f'A rate limit error occured {err_rate}.')
                retry_num += 1
                if retry_num > max_retries:
                    return {'completed': False, 'data': str(err_rate)}
                
                time.sleep(openai_ratelimit_retry_time)

            except (APIStatusError, APIConnectionError, APITimeoutError) as err:
                logging.error(f'A {str(type(err))} error occured {err}.')
                retry_num += 1

                if retry_num > max_retries:
                    return {'completed': False, 'data': str(err)}
                
                
                if isinstance(err, APIStatusError):
                    err_http:APIStatusError = err
                    if err_http.status_code not in retry_code_list:
                        return {'completed': False, 'data': str(err_http)}
                
                delay = calculate_delay(retry_num, delay, exp_base, jitter)
                time.sleep(delay)

            except APIError as err_api:
                logging.error(f'An API error occured {err_api}')
                return {'completed': False, 'data': str(err_api)}
            
        return {'completed': False, 'data': 'An Unknown error occured when accessing OpenAI API'}
    
    return wrapper



