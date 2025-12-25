from pathlib import Path
import sys

# Add AutoCD parent directory (the 'methods' folder) to sys.path
THIS_FILE = Path(__file__).resolve()
AUTOCD_PARENT = THIS_FILE.parent  # .../experiments/graph_discovery/methods
if str(AUTOCD_PARENT) not in sys.path:
    sys.path.insert(0, str(AUTOCD_PARENT))

# Initialize JPype JVM for Tetrad (Java-based causal discovery library)
# This must be done before importing Tetrad classes
try:
    import jpype
    import os
    import subprocess
    
    # Auto-detect JAVA_HOME if not set
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        # Try common Homebrew paths
        homebrew_paths = [
            "/opt/homebrew/opt/openjdk@11",  # Apple Silicon
            "/usr/local/opt/openjdk@11",     # Intel Mac
            "/opt/homebrew/opt/openjdk",
            "/usr/local/opt/openjdk",
        ]
        for brew_path in homebrew_paths:
            if Path(brew_path).exists():
                java_home = brew_path
                os.environ["JAVA_HOME"] = java_home
                # Add to PATH if not already there
                java_bin = f"{java_home}/bin"
                if java_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{java_bin}:{os.environ.get('PATH', '')}"
                break
        
        # Try /usr/libexec/java_home on macOS
        if not java_home:
            try:
                result = subprocess.run(
                    ["/usr/libexec/java_home", "-v", "11"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    java_home = result.stdout.strip()
                    os.environ["JAVA_HOME"] = java_home
            except Exception:
                pass
    
    if not jpype.isJVMStarted():
        jar_dir = AUTOCD_PARENT / "AutoCD" / "jar_files"
        # Collect all jar files
        jar_files = list(jar_dir.glob("*.jar"))
        if jar_files:
            jar_paths = [str(jar) for jar in jar_files]
            jpype.startJVM("-ea", classpath=jar_paths, convertStrings=False)
    # Only import jpype.imports AFTER JVM is started
    import jpype.imports
except Exception as e:
    # If JPype is not available or JVM fails to start, Tetrad algorithms won't work
    # This will be caught later when trying to use Tetrad
    pass
    
from AutoCD.causal_discovery.class_causal_config import CausalConfigurator
from AutoCD.causal_discovery.select_with_OCT_parallel import OCT_parallel
from AutoCD.data_utils.class_data_object import data_object

def autocd_entry(
    samples,
    dataset_name: str = "orca_dataset",
    target_name: str | None = None,
    causal_sufficiency: bool = False,
    n_lags: int | None = None,
    tiers=None,
    n_jobs: int = 2,
):
    """
    AutoCD entrypoint for ORCA experiments.

    Parameters
    ----------
    samples :
        Input data. Either a pandas DataFrame or a 2D numpy array.
    dataset_name : str, optional
        Name used internally by AutoCD for the dataset.
    target_name : str or None, optional
        Optional target name. For generic graph discovery this can be left as None.
    causal_sufficiency : bool, optional
        Whether to restrict to algorithms that assume causal sufficiency.
    n_lags : int or None, optional
        Number of previous time lags for time-series data. Leave as None for
        cross-sectional data (is_time_series=False).
    tiers :
        Optional tier information for Tetrad.
    n_jobs : int, optional
        Number of parallel jobs for the OCT tuning method.

    Returns
    -------
    dict
        A dictionary containing at minimum:

        - \"adjacency\": numpy array adjacency matrix of the selected causal graph
        - \"variables\": list of variable names corresponding to the rows/columns
        - \"optimal_config\": the selected causal configuration dictionary
    """
    import numpy as np
    import pandas as pd

    if isinstance(samples, pd.DataFrame):
        df = samples.copy()
    else:
        arr = np.asarray(samples)
        if arr.ndim != 2:
            raise ValueError("samples must be a 2D array or pandas DataFrame")
        n, d = arr.shape
        cols = [f"V{i}" for i in range(d)]
        df = pd.DataFrame(arr, columns=cols)

    # Build AutoCD data object (handles data types, encoding, time-series flag)
    data_obj = data_object(df, dataset_name, target_name, n_lags)

    # Create causal configurations compatible with this data (cross-sectional vs time-series)
    causal_configs = CausalConfigurator().create_causal_configs(
        data_obj,
        causal_sufficiency,
    )
    if not causal_configs:
        raise ValueError("AutoCD: no causal configurations generated for the given data.")

    # Run OCT tuning to select the best configuration and corresponding graph
    library_results, mec_graph_pd, graph_pd, optimal_config = OCT_parallel(
        data_obj,
        n_jobs,
        tiers=tiers,
    ).select(causal_configs)

    # Convert the selected causal graph to a simple adjacency matrix + variable list
    A = graph_pd.to_numpy()
    variables = list(graph_pd.columns)

    return {
        "adjacency": A,
        "variables": variables,
        "optimal_config": optimal_config,
        "library_results": library_results,
        "mec_graph": mec_graph_pd,
    }