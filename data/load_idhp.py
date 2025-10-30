# load_ihdp.py

import os
import requests
import zipfile
import numpy as np

def download_ihdp(save_dir: str,
                  url: str = "http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip",
                  force_download: bool = False) -> str:
    """
    IHDP 데이터셋을 다운로드하고 압축을 해제한 파일 경로를 반환합니다.

    Parameters
    ----------
    save_dir : str
        데이터를 저장할 디렉토리 경로.
    url : str
        다운로드할 ZIP 파일의 URL.
    force_download : bool
        이미 파일이 존재해도 강제로 다시 다운로드할지 여부.

    Returns
    -------
    npz_path : str
        압축 해제된 .npz 파일의 경로.
    """
    os.makedirs(save_dir, exist_ok=True)

    zip_filename = os.path.basename(url)
    zip_path = os.path.join(save_dir, zip_filename)
    npz_filename = zip_filename.replace(".zip", "")
    npz_path = os.path.join(save_dir, npz_filename)

    # 다운로드
    if force_download or not os.path.exists(npz_path):
        print(f"'{url}' 에서 다운로드 시도 …")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 파일 저장
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"다운로드 완료: {zip_path}")

        # 압축 해제
        print(f"압축 해제 중: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(save_dir)
        print(f"압축 해제 완료. {npz_path} 확인하세요.")

    else:
        print(f"이미 존재함: {npz_path}")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f".npz 파일을 찾을 수 없습니다: {npz_path}")

    return npz_path

def load_ihdp(npz_path: str) -> dict:
    """
    .npz 파일을 읽어 들여 데이터 딕셔너리 형태로 반환합니다.

    Parameters
    ----------
    npz_path : str
        .npz 파일 경로.

    Returns
    -------
    data_dict : dict
        키: ['x', 't', 'yf', 'ycf', 'ite', 'mu0', 'mu1'] 예상
    """
    print(f"파일 읽는 중: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    # 예상되는 키 보여주기
    print("데이터 키:", data.files)
    return {key: data[key] for key in data.files}

def main():
    # 사용 예시
    save_dir = "data/raw/ihdp"
    url = "http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip"

    npz_path = download_ihdp(save_dir=save_dir, url=url, force_download=False)
    data = load_ihdp(npz_path)

    # 간단히 shape 확인
    print("x shape:", data.get("x", None).shape)
    # 예: 첫 번째 반복의 ATE 계산
    if "ite" in data:
        print("ite shape:", data["ite"].shape)
        ate0 = data["ite"][:, 0].mean()
        print(f"첫 번째 시뮬레이션의 평균 treatment effect (ATE): {ate0:.4f}")
    else:
        print("Note: 'ite' 키가 없습니다. available keys:", data.keys())
        # 예: ymul 키가 있다면 사용
        if "ymul" in data:
            print("ymul shape:", data["ymul"].shape)
            # ymul의 구조 파악 후 ATE 계산 가능성 고려
        else:
            print("적절한 treatment effect 키를 찾아야 합니다.")

if __name__ == "__main__":
    main()