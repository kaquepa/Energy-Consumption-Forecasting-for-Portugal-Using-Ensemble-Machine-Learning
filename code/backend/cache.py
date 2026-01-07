import time

class TTLCache:
    def __init__(self):
        self.store = {}
    def get(self, key):
        item = self.store.get(key)
        if not item:
            return None
        value, expires = item
        if time.time() > expires:
            del self.store[key]
            return None
        return value
    
    def set(self, key, value, ttl):
        self.store[key] = (value, time.time() + ttl)
cache = TTLCache()
