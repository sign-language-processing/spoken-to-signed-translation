from collections import OrderedDict


class LRUCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end to show it's recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            # Move the accessed item to the end to show it's recently used
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # Remove the first (least recently used) item
            self.cache.popitem(last=False)
        self.cache[key] = value
