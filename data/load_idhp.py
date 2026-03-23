# load_ihdp.py

import os
import requests
import zipfile
import numpy as np

def download_ihdp(save_dir: str,
                  url: str = "http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip",
                  force_download: bool = False) -> str:
    """
    Download the IHDP dataset and return the path to the extracted file.

    Parameters
    ----------
    save_dir : str
        Directory where data will be saved.
    url : str
        URL of the ZIP file to download.
    force_download : bool
        Re-download even if the file already exists.

    Returns
    -------
    npz_path : str
        Path to the extracted .npz file.
    """
    os.makedirs(save_dir, exist_ok=True)

    zip_filename = os.path.basename(url)
    zip_path = os.path.join(save_dir, zip_filename)
    npz_filename = zip_filename.replace(".zip", "")
    npz_path = os.path.join(save_dir, npz_filename)

    # download
    if force_download or not os.path.exists(npz_path):
        print(f"Downloading from '{url}' …")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # save file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Download complete: {zip_path}")

        # extract archive
        print(f"Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(save_dir)
        print(f"Extraction complete. See {npz_path}")

    else:
        print(f"Already exists: {npz_path}")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f".npz file not found: {npz_path}")

    return npz_path

def load_ihdp(npz_path: str) -> dict:
    """
    Load an .npz file and return it as a data dictionary.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file.

    Returns
    -------
    data_dict : dict
        Expected keys: ['x', 't', 'yf', 'ycf', 'ite', 'mu0', 'mu1']
    """
    print(f"Reading file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    # show expected keys
    print("Data keys:", data.files)
    return {key: data[key] for key in data.files}

def main():
    # usage example
    save_dir = "data/raw/ihdp"
    url = "http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip"

    npz_path = download_ihdp(save_dir=save_dir, url=url, force_download=False)
    data = load_ihdp(npz_path)

    # quick shape check
    print("x shape:", data.get("x", None).shape)
    # e.g. compute ATE for first iteration
    if "ite" in data:
        print("ite shape:", data["ite"].shape)
        ate0 = data["ite"][:, 0].mean()
        print(f"Mean treatment effect (ATE) for first simulation: {ate0:.4f}")
    else:
        print("Note: 'ite' key not found. Available keys:", data.keys())
        # e.g. use ymul key if available
        if "ymul" in data:
            print("ymul shape:", data["ymul"].shape)
            # inspect ymul structure before computing ATE
        else:
            print("No suitable treatment effect key found.")

if __name__ == "__main__":
    main()