# core/llm_metrics.py
import time
import functools
from typing import Dict, Any, Optional, Callable
from ..metrics.collector import MetricsCollector, MetricType, get_metrics_collector

def track_llm_call(agent_name: str, model_name: Optional[str] = None):
    """Decorator to track LLM API calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            if not collector:
                return func(*args, **kwargs)
            
            start_time = time.time()
            start_tokens = 0  # Will be updated if available
            
            try:
                result = func(*args, **kwargs)
                
                # Extract token count if available
                token_count = 0
                if hasattr(result, 'usage'):
                    if hasattr(result.usage, 'total_tokens'):
                        token_count = result.usage.total_tokens
                    elif hasattr(result.usage, 'prompt_tokens') and hasattr(result.usage, 'completion_tokens'):
                        token_count = result.usage.prompt_tokens + result.usage.completion_tokens
                
                # Record metrics
                duration = time.time() - start_time
                metadata = {
                    "model": model_name or "unknown",
                    "function": func.__name__,
                    "success": True
                }
                
                collector.record_api_call(agent_name, func.__name__, duration, metadata)
                if token_count > 0:
                    collector.record_token_count(agent_name, token_count, metadata)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metadata = {
                    "model": model_name or "unknown",
                    "function": func.__name__,
                    "success": False,
                    "error": str(e)
                }
                
                collector.record_api_call(agent_name, func.__name__, duration, metadata)
                collector.record_error(agent_name, type(e).__name__, metadata)
                raise
                
        return wrapper
    return decorator

def track_llm_generation(agent_name: str, model_name: Optional[str] = None):
    """Decorator specifically for text generation calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            if not collector:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Extract generation metrics
                duration = time.time() - start_time
                metadata = {
                    "model": model_name or "unknown",
                    "function": func.__name__,
                    "type": "generation",
                    "success": True
                }
                
                # Try to extract token information
                if hasattr(result, 'usage'):
                    if hasattr(result.usage, 'total_tokens'):
                        token_count = result.usage.total_tokens
                        collector.record_token_count(agent_name, token_count, metadata)
                    elif hasattr(result.usage, 'prompt_tokens') and hasattr(result.usage, 'completion_tokens'):
                        prompt_tokens = result.usage.prompt_tokens
                        completion_tokens = result.usage.completion_tokens
                        total_tokens = prompt_tokens + completion_tokens
                        
                        collector.record_token_count(agent_name, total_tokens, metadata)
                        collector.record_metric(
                            MetricType.CUSTOM, 
                            prompt_tokens, 
                            agent_name, 
                            {**metadata, "token_type": "prompt"}
                        )
                        collector.record_metric(
                            MetricType.CUSTOM, 
                            completion_tokens, 
                            agent_name, 
                            {**metadata, "token_type": "completion"}
                        )
                
                collector.record_api_call(agent_name, func.__name__, duration, metadata)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metadata = {
                    "model": model_name or "unknown",
                    "function": func.__name__,
                    "type": "generation",
                    "success": False,
                    "error": str(e)
                }
                
                collector.record_api_call(agent_name, func.__name__, duration, metadata)
                collector.record_error(agent_name, type(e).__name__, metadata)
                raise
                
        return wrapper
    return decorator

class LLMMetricsTracker:
    """Context manager for tracking LLM operations"""
    
    def __init__(self, agent_name: str, operation: str, model_name: Optional[str] = None):
        self.agent_name = agent_name
        self.operation = operation
        self.model_name = model_name
        self.collector = get_metrics_collector()
        self.start_time = None
        self.start_tokens = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.collector or not self.start_time:
            return
            
        duration = time.time() - self.start_time
        metadata = {
            "model": self.model_name or "unknown",
            "operation": self.operation,
            "success": exc_type is None
        }
        
        if exc_type:
            metadata["error"] = str(exc_val)
            self.collector.record_error(self.agent_name, exc_type.__name__, metadata)
        
        self.collector.record_api_call(self.agent_name, self.operation, duration, metadata)
        
    def record_tokens(self, token_count: int, token_type: str = "total"):
        """Record token count for the current operation"""
        if self.collector:
            metadata = {
                "model": self.model_name or "unknown",
                "operation": self.operation,
                "token_type": token_type
            }
            self.collector.record_token_count(self.agent_name, token_count, metadata)

# Example usage functions
def create_llm_tracker(agent_name: str, operation: str, model_name: Optional[str] = None) -> LLMMetricsTracker:
    """Create an LLM metrics tracker"""
    return LLMMetricsTracker(agent_name, operation, model_name)

def record_llm_tokens(agent_name: str, token_count: int, model_name: Optional[str] = None, token_type: str = "total"):
    """Record token count for LLM operations"""
    collector = get_metrics_collector()
    if collector:
        metadata = {
            "model": model_name or "unknown",
            "token_type": token_type
        }
        collector.record_token_count(agent_name, token_count, metadata)
