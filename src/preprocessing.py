import os
import re
import numpy as np
import nibabel as nib # Used to open .hdr volumes 

from data_download import download_oasis

from glob import glob # Used to match characters when looking up files/dir 
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm # Used to show progress bar when analyzing subjects 
import json # Used to save patient dataset
import shutil # Used to cleanup raw directory after preprocessing

'''
    Reads from the patient profile the CDR value
    and converts it into a label.
'''
def parse_metadata(txt_path):    
    with open(txt_path, "r") as f:
        text = f.read()

    # Capture the CDR value within square brackets 
    match = re.search(r"CDR:\s*([0-9.]*)", text)
    if not match:
        print(f"Couldnt find CDR value in {txt_path}")
        return None

    # Extract the CDR value from the match 
    cdr_str = match.group(1).strip()
    if cdr_str == "":
        #print(f"CDR is blank in {txt_path}")
        return None  # Missing value

    cdr = float(cdr_str)
    '''
        Usually Alzheimer is indicated by CDR = 1.
        Intermediate values such greater and equal
        to CDR = 0.5 will be considered early onset 
        and will be assigned the same label.
    '''
    label = 1 if cdr >= 0.5 else 0  
    return label

'''
    Use nibabel to load a .hrd/.img volume. 
    Choose the first volume availabe for a given
    patient.
'''
def load_hdr_volume(subject_raw_dir):
    hdr_files = glob(os.path.join(subject_raw_dir, "*.hdr"))
    if not hdr_files:
        print(f"Couldn't find hdr file in {subject_raw_dir}")
        return None

    hdr_path = hdr_files[0]
    try:
        img = nib.load(hdr_path)
        volume = img.get_fdata()
        return volume
    except Exception as e:
        print(f"Something went wrong when loading the hdr volume in {subject_raw_dir}")
        print(e)
        return None

'''
    Normalize and resample the 3D volume.
'''
def preprocess_volume(volume, target_shape=(128, 128, 128)):
    volume = volume.astype(np.float32)

    # Move channel dimension in the 1st first place
    # to satisfy torch requirements. 
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = np.moveaxis(volume, -1, 0)   
    elif volume.ndim == 3:
        volume = volume[None, ...]            
        
    # Resample 3D (apply zoom only on spatial dims)
    # volume has shape (1, D, H, W), so zoom only on the last 3 dims
    zoom_factors = (
        1,  # don't zoom the channel dimension
        target_shape[0] / volume.shape[1],
        target_shape[1] / volume.shape[2],
        target_shape[2] / volume.shape[3],
    )
    volume_resampled = zoom(volume, zoom_factors)

    # Normalize intensity
    volume_resampled = (volume_resampled - np.mean(volume_resampled)) / (
                        np.std(volume_resampled) + 1e-6)
    
    return volume_resampled

def preprocess_all(n_discs=2, data_raw_dir="./data/raw", processed_dir="./data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    
    dataset_final = []
    
    subjects = []
    for disc_id in range(1, n_discs+1):        
        disc_dataset_path = Path(processed_dir) / f"disc{disc_id}.json"   
        disc_dataset = []
        
        # Skip data already preprocessed 
        if disc_dataset_path.exists():
            with open(disc_dataset_path, "r") as f:
                disc_dataset = json.load(f)
                                
            dataset_final.extend(disc_dataset) 
            
            print(f"disc{disc_id} already fully processed, skipping.")
            continue
            
        download_oasis(disc_id, Path(data_raw_dir))
        
        disc_dir = Path(data_raw_dir) / f"disc{disc_id}"        
        subjects = sorted([d for d in Path(disc_dir).glob("OAS1_*") if d.is_dir()])            

        for subject_dir in tqdm(subjects, desc=f"Processing subjects from disc{disc_id}"):
            session_name = subject_dir.name  # Ex. OAS1_0039_MR1
            txt_path = subject_dir / f"{session_name}.txt"
            
            out_path = Path(processed_dir) / f"{session_name}.npy"
            
            # Parse metadata to get label
            label = parse_metadata(txt_path)
            if label is None:
                #print(f"\nCouldn't extract label for patient {session_name}")
                continue
            
            # Add entry to dataset index
            disc_dataset.append({
                "id": session_name,
                "path": str(out_path),
                "label": label
            })            
            
            if not txt_path.exists():
                print(f"\nCouldn't find the txt file for patient {session_name}")
                continue

            # RAW folder
            raw_dir = subject_dir / "RAW"
            if not raw_dir.exists():
                print(f"\nCouldn't find raw folder for patient {session_name}")
                continue

            # Load HDR volume
            volume = load_hdr_volume(raw_dir)
            if volume is None:
                print(f"\nCouldn't load hdr volume for patient {session_name}")
                continue

            # Save preprocessed volume 
            np.save(out_path, preprocess_volume(volume))

        dataset_final.extend(disc_dataset)
            
        # Save partial dataset
        with open(disc_dataset_path, "w") as f:
            json.dump(disc_dataset, f, indent=2)
                    
        # Cleanup of raw files 
        if disc_dir.exists():
            shutil.rmtree(disc_dir)
            
    # Save dataset.json
    with open(Path(processed_dir) / "dataset.json", "w") as f:
        json.dump(dataset_final, f, indent=2)
        
    print(f"Done! Saved {len(dataset_final)} subjects.")


if __name__ == "__main__":
    preprocess_all()
