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
        "__import__", "open", "exec", "eval", "compile", "input", "raw_input",
        "file", "reload", "vars", "globals", "locals", "dir", "hasattr",
        "getattr", "setattr", "delattr", "callable", "isinstance", "issubclass",
        "type", "super", "property", "staticmethod", "classmethod", "abs",
        "all", "any", "bin", "bool", "bytearray", "bytes", "chr", "ord",
        "complex", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "hex", "int", "iter", "len", "list", "map", "max",
        "min", "next", "oct", "pow", "print", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip"
    }
    
    # Allowed modules (whitelist approach)
    ALLOWED_MODULES = {
        # Data manipulation
        "pandas", "numpy", "scipy", "sklearn", "statsmodels",
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
    MAX_MEMORY_MB = 512  # MB
    MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB
    MAX_CODE_LENGTH = 10000  # characters
    
    def __init__(self, 
                 max_execution_time: float = MAX_EXECUTION_TIME,
                 max_memory_mb: int = MAX_MEMORY_MB,
                 max_output_size: int = MAX_OUTPUT_SIZE,
                 max_code_length: int = MAX_CODE_LENGTH):
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.max_code_length = max_code_length
        
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
                if alias.name not in self.ALLOWED_MODULES:
                    raise SecurityViolationError(f"Module not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in self.ALLOWED_MODULES:
                raise SecurityViolationError(f"Module not allowed: {node.module}")
    
    @contextmanager
    def _resource_limits(self):
        """Set resource limits for code execution"""
        # Set memory limit
        if hasattr(resource, 'RLIMIT_AS'):
            resource.setrlimit(resource.RLIMIT_AS, 
                             (self.max_memory_mb * 1024 * 1024, -1))
        
        # Set CPU time limit
        if hasattr(resource, 'RLIMIT_CPU'):
            resource.setrlimit(resource.RLIMIT_CPU, (int(self.max_execution_time), -1))
        
        try:
            yield
        finally:
            # Reset limits
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
            if hasattr(resource, 'RLIMIT_CPU'):
                resource.setrlimit(resource.RLIMIT_CPU, (-1, -1))
    
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
