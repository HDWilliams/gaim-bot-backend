import time
from collections import deque
from typing import Tuple

class RateLimiter:
    def __init__(self, rate_limit=100, period=1, token_limit=30000):
        self.rate_limit = rate_limit
        self.period = period
        self.token_limit = token_limit
        self.current_requests = deque([])
    
    def enforce_rate_limit(self) -> None:
        """ Used with each request to maintain ratelimit for LLM api"""
        while len(self.current_requests) >= self.rate_limit:
            now = time.time()
            oldest_request = self.current_requests[0]
            if now - oldest_request > self.period:
                self.current_requests.popleft()
            else:
                delay = (oldest_request + self.period) - now
                time.sleep(delay)
                
        self.current_requests.append(time.time())
