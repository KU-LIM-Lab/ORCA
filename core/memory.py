# core/memory.py
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import json
import hashlib
from collections import OrderedDict

class MemoryItem:
    """Individual memory item"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.accessed_at = datetime.now()
        self.access_count = 0
        self.ttl = ttl  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if memory item is expired"""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)
    
    def access(self) -> Any:
        """Access the memory item"""
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl": self.ttl
        }

class AgentMemory:
    """Memory management for agents"""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.memory: OrderedDict[str, MemoryItem] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory"""
        if key in self.memory:
            item = self.memory[key]
            if item.is_expired():
                del self.memory[key]
                self.miss_count += 1
                return None
            
            # Move to end (most recently used)
            self.memory.move_to_end(key)
            self.hit_count += 1
            return item.access()
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory"""
        # Remove expired items
        self._cleanup_expired()
        
        # Remove oldest items if at capacity
        while len(self.memory) >= self.max_size:
            self.memory.popitem(last=False)
        
        # Set new item
        item_ttl = ttl if ttl is not None else self.default_ttl
        self.memory[key] = MemoryItem(key, value, item_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from memory"""
        if key in self.memory:
            del self.memory[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memory"""
        self.memory.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def _cleanup_expired(self) -> None:
        """Remove expired items"""
        expired_keys = [key for key, item in self.memory.items() if item.is_expired()]
        for key in expired_keys:
            del self.memory[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.memory),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "memory_usage": sum(len(str(item.value)) for item in self.memory.values())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary"""
        return {
            "items": {key: item.to_dict() for key, item in self.memory.items()},
            "stats": self.get_stats()
        }

class SessionMemory:
    """Session-based memory management"""
    
    def __init__(self):
        self.sessions: Dict[str, AgentMemory] = {}
        self.default_max_size = 1000
        self.default_ttl = 3600  # 1 hour
    
    def get_session(self, session_id: str) -> AgentMemory:
        """Get or create session memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = AgentMemory(
                max_size=self.default_max_size,
                default_ttl=self.default_ttl
            )
        return self.sessions[session_id]
    
    def get(self, session_id: str, key: str) -> Optional[Any]:
        """Get value from session memory"""
        session = self.get_session(session_id)
        return session.get(key)
    
    def set(self, session_id: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in session memory"""
        session = self.get_session(session_id)
        session.set(key, value, ttl)
    
    def delete(self, session_id: str, key: str) -> bool:
        """Delete value from session memory"""
        if session_id in self.sessions:
            return self.sessions[session_id].delete(key)
        return False
    
    def clear_session(self, session_id: str) -> None:
        """Clear session memory"""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions"""
        return {
            session_id: session.get_stats()
            for session_id, session in self.sessions.items()
        }

# Global session memory
session_memory = SessionMemory()

# Utility functions
def generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()

def cache_result(ttl: Optional[int] = None):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = session_memory.get("default", cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            session_memory.set("default", cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator
