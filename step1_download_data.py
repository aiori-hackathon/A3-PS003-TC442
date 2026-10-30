"""
STEP 1 - Download the UCI Traffic Flow Forecasting dataset.

    python step1_download_data.py

Downloads the ZIP from the UCI repository and extracts traffic_dataset.mat
into data/raw/. If the download fails (firewall, proxy, SSL), download the ZIP
manually from https://archive.ics.uci.edu/dataset/608/traffic+flow+forecasting
and put traffic_dataset.mat into data/raw/ yourself.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

from src import config


def download(url: str, dest: Path) -> None:
    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  downloaded {done / 1e6:5.2f} / {total / 1e6:.2f} MB", end="")

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main() -> int:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if config.MAT_FILE.exists():
        print(f"Already present: {config.MAT_FILE} - nothing to do.")
        return 0

    zip_path = config.DATA_RAW / "traffic_flow_forecasting.zip"
    print(f"Downloading {config.DATASET_URL}")
    try:
        download(config.DATASET_URL, zip_path)
    except Exception as e:  # network problems are the usual cause
        print(f"\nDownload failed: {e}")
        print("Download the ZIP manually (see the top of this file) and place "
              "traffic_dataset.mat in data/raw/.")
        return 1

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist()
                       if m.endswith("traffic_dataset.mat") and "__MACOSX" not in m]
            if not members:
                print(f"traffic_dataset.mat not found in the ZIP. Contents: {zf.namelist()}")
                return 1
            with zf.open(members[0]) as src, open(config.MAT_FILE, "wb") as dst:
                dst.write(src.read())
            for m in zf.namelist():  # keep the dataset's own README too
                if m.endswith(".docx") and "__MACOSX" not in m:
                    (config.DATA_RAW / Path(m).name).write_bytes(zf.read(m))
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid ZIP. Delete it and try again.")
        return 1
    finally:
        zip_path.unlink(missing_ok=True)

    size_mb = config.MAT_FILE.stat().st_size / 1e6
    print(f"Saved {config.MAT_FILE} ({size_mb:.1f} MB)")
    print("Next: python step2_inspect_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
