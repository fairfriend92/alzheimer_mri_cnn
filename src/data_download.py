import urllib.request
import tarfile
from pathlib import Path

def download_oasis(disc_id, destination=Path("./data/raw")):   
    destination.mkdir(parents=True, exist_ok=True)    
    
    # This file contains several preprocessed MRI scans (~1.3 GB)
    url = "https://download.nrg.wustl.edu/data/oasis_cross-sectional_"+f"disc{disc_id}.tar.gz"

    archive_path = destination / f"oasis_{disc_id}.tar.gz"

    if archive_path.exists():
        print(f"Archive {disc_id} already downloaded.")
    else:
        print(f"Downloading OASIS {disc_id} dataset...")
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:
            print("Error downloading dataset from " + str(url))
            print(e)
            return

        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=destination)

        # Remove the .tar.gz archive
        archive_path.unlink()     

if __name__ == "__main__":
    download_oasis()
