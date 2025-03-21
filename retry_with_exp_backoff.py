from openai import RateLimitError, APIStatusError, APIConnectionError, APIError
import random
import time

from RateLimiter import RateLimiter

def retry_with_exp_backoff(func, rate_limiter:RateLimiter=RateLimiter(), jitter:bool=True, 
                           init_delay:float=1, 
                           exp_base:int=2,
                           max_retries:int=4, 
                           retry_code_list = (429, 500, 502, 503, 504)
                           ):
    def wrapper(*args, **kwargs):

        num_retries = 0
        while num_retries < max_retries:
            try:
                rate_limiter.enforce_rate_limit()
                return func(*args, **kwargs)
            except (APIStatusError, RateLimitError, APIConnectionError) as err:

                if err.status_code in retry_code_list:
                    num_retries += 1
                    delay = exp_base**(num_retries - 1) * init_delay
                    if jitter:
                        delay += random.uniform(0,1)
                    time.sleep(delay)
                else:
                    return {'status_code': err.status_code}
            except APIError as err:
                return {'error': str(err)}
    
    return wrapper



