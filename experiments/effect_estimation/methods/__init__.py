# Import from parent directory's methods.py file directly
# methods.py is in experiments/effect_estimation/, not in methods/
import importlib.util
from pathlib import Path

# Get the parent directory (experiments/effect_estimation)
parent_dir = Path(__file__).parent.parent
methods_py_path = parent_dir / "methods.py"

# Load methods.py as a module
spec = importlib.util.spec_from_file_location("methods", methods_py_path)
methods_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(methods_module)

# Export the functions
get_method = methods_module.get_method
register_method = methods_module.register_method
METHOD_REGISTRY = methods_module.METHOD_REGISTRY

__all__ = ['get_method', 'register_method', 'METHOD_REGISTRY']

