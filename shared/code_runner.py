# shared/code_runner.py
import builtins
import time
import signal
import sys
import os
import ast
import subprocess
import tempfile
import resource
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
import logging

# Configure logging
logger = logging.getLogger(__name__)

class CodeExecutionError(Exception):
    """Base exception for code execution errors"""
    pass

class SecurityViolationError(CodeExecutionError):
    """Raised when code violates security restrictions"""
    pass

class ResourceLimitError(CodeExecutionError):
    """Raised when code exceeds resource limits"""
    pass

class TimeoutError(CodeExecutionError):
    """Raised when code execution times out"""
    pass

class CodeRunner:
    """Safe Python code execution with comprehensive security measures"""
    
    # Forbidden builtins and functions
    FORBIDDEN_BUILTINS = {
        "open", "exec", "eval", "compile", "input", "raw_input",
        "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr",
    }
    
    
    # Allowed modules (whitelist approach)
    ALLOWED_MODULES = {
        # Data manipulation
        "pandas", "numpy", "scipy", "sklearn", "statsmodels", "networkx",
        # Causal discovery
        "lingam", "pgmpy", 
        # Causal inference
         "dowhy", "causallearn",
        # Visualization
        "matplotlib", "seaborn", "plotly", "bokeh",
        # Statistics
        "scipy.stats", "statsmodels.api", "statsmodels.stats",
        # Math
        "math", "cmath", "decimal", "fractions",
        # JSON/CSV
        "json", "csv", "io",
        # Date/Time
        "datetime", "time", "calendar",
        # System (limited)
        "os.path", "sys.version", "platform"
    }
    
    # Resource limits
    MAX_EXECUTION_TIME = 30.0  # seconds
    MAX_MEMORY_MB = 8192  # MB (increased from 512)
    MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB
    MAX_CODE_LENGTH = 10000  # characters
    
    def __init__(self, 
                 max_execution_time: float = MAX_EXECUTION_TIME,
                 max_memory_mb: int = MAX_MEMORY_MB,
                 max_output_size: int = MAX_OUTPUT_SIZE,
                 max_code_length: int = MAX_CODE_LENGTH,
                 allow_all_imports: bool = True):
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.max_code_length = max_code_length
        self.allow_all_imports = allow_all_imports
        
    def validate_code(self, code: str) -> None:
        """Validate code for security and syntax issues"""
        if len(code) > self.max_code_length:
            raise SecurityViolationError(f"Code too long: {len(code)} > {self.max_code_length}")
        
        try:
            # Parse AST to check for forbidden constructs
            tree = ast.parse(code)
            self._check_ast_security(tree)
        except SyntaxError as e:
            raise CodeExecutionError(f"Syntax error: {e}")
    
    def _check_ast_security(self, tree: ast.AST) -> None:
        """Check AST for security violations"""
        for node in ast.walk(tree):
            # Check for forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_BUILTINS:
                        raise SecurityViolationError(f"Forbidden function: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous attribute access
                    if self._is_dangerous_attribute(node.func):
                        raise SecurityViolationError(f"Dangerous attribute access: {ast.unparse(node.func)}")
            
            # Check for imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_import_security(node)
            
            # Check for exec/eval calls
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id in ["exec", "eval"]:
                            raise SecurityViolationError("exec/eval calls not allowed")
    
    def _is_dangerous_attribute(self, attr: ast.Attribute) -> bool:
        """Check if attribute access is dangerous"""
        dangerous_attrs = {
            "__import__", "__globals__", "__locals__", "__code__", "__closure__",
            "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr"
        }
        return attr.attr in dangerous_attrs
    
    def _check_import_security(self, node: ast.AST) -> None:
        """Check import security"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                # Check if module or its parent is allowed
                if not self._is_module_allowed(module_name):
                    raise SecurityViolationError(f"Module not allowed: {module_name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module
                # Check if module or its parent is allowed
                if not self._is_module_allowed(module_name):
                    raise SecurityViolationError(f"Module not allowed: {module_name}")
    
    def _is_module_allowed(self, module_name: str) -> bool:
        """Check if a module (or its parent) is in the allowed list"""
        # Exact match
        if module_name in self.ALLOWED_MODULES:
            return True
        
        # Check if parent module is allowed (e.g., "causal_learn.algorithms" -> "causal_learn")
        parts = module_name.split('.')
        for i in range(1, len(parts)):
            parent_module = '.'.join(parts[:i])
            if parent_module in self.ALLOWED_MODULES:
                return True
        
        return False
    
    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Safe wrapper for __import__ that checks whitelist (or allows all if allow_all_imports=True)"""
        # If allow_all_imports is True, skip the whitelist check
        if not self.allow_all_imports:
            # Check if the module is allowed
            if not self._is_module_allowed(name):
                raise SecurityViolationError(f"Module not allowed: {name}")
        
        # Use the real __import__
        return __import__(name, globals, locals, fromlist, level)
    
    @contextmanager
    def _resource_limits(self):
        """Set resource limits for code execution"""
        memory_limit_set = False
        cpu_limit_set = False
        
        # Set memory limit (only if safe to do so)
        if hasattr(resource, 'RLIMIT_AS'):
            try:
                current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
                desired_limit = self.max_memory_mb * 1024 * 1024
                # Only set if current hard limit allows it, or if we want unlimited
                if current_hard == -1 or desired_limit <= current_hard:
                    resource.setrlimit(resource.RLIMIT_AS, (desired_limit, current_hard))
                    memory_limit_set = True
                else:
                    logger.warning(f"Memory limit {self.max_memory_mb}MB exceeds system limit, skipping")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not set memory limit: {e}, skipping")
        
        # Set CPU time limit (only if safe to do so)
        if hasattr(resource, 'RLIMIT_CPU'):
            try:
                current_soft, current_hard = resource.getrlimit(resource.RLIMIT_CPU)
                desired_limit = int(self.max_execution_time)
                # Only set if current hard limit allows it, or if we want unlimited
                if current_hard == -1 or desired_limit <= current_hard:
                    resource.setrlimit(resource.RLIMIT_CPU, (desired_limit, current_hard))
                    cpu_limit_set = True
                else:
                    logger.warning(f"CPU time limit {self.max_execution_time}s exceeds system limit, skipping")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not set CPU time limit: {e}, skipping")
        
        try:
            yield
        finally:
            # Reset limits only if we set them
            if memory_limit_set and hasattr(resource, 'RLIMIT_AS'):
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
                except (ValueError, OSError):
                    pass  # Ignore errors when resetting
            if cpu_limit_set and hasattr(resource, 'RLIMIT_CPU'):
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (-1, -1))
                except (ValueError, OSError):
                    pass  # Ignore errors when resetting
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout signal"""
        raise TimeoutError("Code execution timed out")
    
    def execute(self, 
                code: str, 
                globals_dict: Optional[Dict[str, Any]] = None,
                locals_dict: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Execute Python code safely
        
        Args:
            code: Python code to execute
            globals_dict: Global variables to provide
            locals_dict: Local variables to provide
            timeout: Execution timeout (overrides default)
            
        Returns:
            Tuple of (globals, locals, output)
            
        Raises:
            SecurityViolationError: If code violates security restrictions
            TimeoutError: If code execution times out
            ResourceLimitError: If code exceeds resource limits
            CodeExecutionError: For other execution errors
        """
        # Validate code
        self.validate_code(code)
        
        # Set timeout
        actual_timeout = timeout or self.max_execution_time
        
        # Prepare execution environment
        safe_globals = self._create_safe_globals()
        if globals_dict:
            safe_globals.update(globals_dict)
        
        safe_locals = {}
        if locals_dict:
            safe_locals.update(locals_dict)
        
        # Capture output
        output_capture = []
        
        def custom_print(*args, **kwargs):
            output_capture.append(' '.join(str(arg) for arg in args))
        
        safe_globals['print'] = custom_print
        
        # Execute with resource limits and timeout
        try:
            with self._resource_limits():
                # Set up timeout signal
                old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(int(actual_timeout))
                
                try:
                    start_time = time.time()
                    
                    # Execute code
                    exec(code, safe_globals, safe_locals)
                    
                    execution_time = time.time() - start_time
                    logger.info(f"Code executed successfully in {execution_time:.2f}s")
                    
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
        except TimeoutError:
            raise TimeoutError(f"Code execution timed out after {actual_timeout}s")
        except MemoryError:
            raise ResourceLimitError(f"Code exceeded memory limit of {self.max_memory_mb}MB")
        except Exception as e:
            raise CodeExecutionError(f"Code execution failed: {str(e)}")
        
        # Check output size
        output_str = '\n'.join(output_capture)
        if len(output_str) > self.max_output_size:
            raise ResourceLimitError(f"Output too large: {len(output_str)} > {self.max_output_size}")
        
        return safe_globals, safe_locals, output_str
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global environment"""
        safe_globals = {
            "__builtins__": {
                # Only allow safe builtins
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "frozenset": frozenset,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow,
                "divmod": divmod,
                "all": all,
                "any": any,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "delattr": delattr,
                "callable": callable,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "super": super,
                "type": type,
                "object": object,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "RuntimeError": RuntimeError,
                "NotImplementedError": NotImplementedError,
                "StopIteration": StopIteration,
                "GeneratorExit": GeneratorExit,
                "SystemExit": SystemExit,
                "KeyboardInterrupt": KeyboardInterrupt,
                "AssertionError": AssertionError,
                "ArithmeticError": ArithmeticError,
                "FloatingPointError": FloatingPointError,
                "OverflowError": OverflowError,
                "ZeroDivisionError": ZeroDivisionError,
                "OSError": OSError,
                "IOError": IOError,
                "EnvironmentError": EnvironmentError,
                "EOFError": EOFError,
                "ImportError": ImportError,
                "ModuleNotFoundError": ModuleNotFoundError,
                "NameError": NameError,
                "UnboundLocalError": UnboundLocalError,
                "SyntaxError": SyntaxError,
                "IndentationError": IndentationError,
                "TabError": TabError,
                "UnicodeError": UnicodeError,
                "UnicodeDecodeError": UnicodeDecodeError,
                "UnicodeEncodeError": UnicodeEncodeError,
                "UnicodeTranslateError": UnicodeTranslateError,
                "Warning": Warning,
                "UserWarning": UserWarning,
                "DeprecationWarning": DeprecationWarning,
                "PendingDeprecationWarning": PendingDeprecationWarning,
                "SyntaxWarning": SyntaxWarning,
                "RuntimeWarning": RuntimeWarning,
                "FutureWarning": FutureWarning,
                "ImportWarning": ImportWarning,
                "UnicodeWarning": UnicodeWarning,
                "BytesWarning": BytesWarning,
                "ResourceWarning": ResourceWarning,
                # Provide safe __import__ that checks whitelist
                "__import__": self._safe_import,
            }
        }
        
        # Add allowed modules
        for module_name in self.ALLOWED_MODULES:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                logger.warning(f"Module {module_name} not available")
        
        return safe_globals

# Convenience functions for backward compatibility
def safe_exec_python(code: str, 
                    globals_dict: Optional[Dict[str, Any]] = None,
                    locals_dict: Optional[Dict[str, Any]] = None,
                    timeout: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Convenience function for backward compatibility"""
    runner = CodeRunner()
    return runner.execute(code, globals_dict, locals_dict, timeout)

# Global instance for easy access
default_runner = CodeRunner()

def execute_code(code: str, 
                globals_dict: Optional[Dict[str, Any]] = None,
                locals_dict: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Execute code using default runner"""
    return default_runner.execute(code, globals_dict, locals_dict, timeout)
