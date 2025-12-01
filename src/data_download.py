import urllib.request
import tarfile
from pathlib import Path

def download_oasis(destination=Path("./data/raw"), n_discs=2):   
    destination.mkdir(parents=True, exist_ok=True)
    
    for disc_id in range(1, n_discs+1):        
        # This file contains several preprocessed MRI scans (~1.3 GB)
        url = "https://download.nrg.wustl.edu/data/oasis_cross-sectional_"+f"disc{disc_id}.tar.gz"

        archive_path = destination / f"oasis_{disc_id}.tar.gz"

        if archive_path.exists():
            print(f"Archive {disc_id} already downloaded.")
        else:
            print(f"Downloading OASIS {disc_id} dataset")
            try:
                urllib.request.urlretrieve(url, archive_path)
                print("Download complete.")
            except Exception as e:
                print("Error downloading dataset from " + str(url))
                print(e)
                return

            print("Extracting dataset...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=destination)
            print("Extraction finished.")

if __name__ == "__main__":
    download_oasis()
