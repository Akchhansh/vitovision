import os
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "wengziheng/mcd_rppg"
# Folder name updated to reflect the 600 patients scale
output_dir = "./mcd_rppg_600_patients" 
os.makedirs(output_dir, exist_ok=True)

print("[Step 1] Downloading main master index (db.csv)...")
db_path = hf_hub_download(
    repo_id=repo_id, 
    filename="db.csv", 
    repo_type="dataset", 
    local_dir=output_dir
)

# Read the database file using pandas
df = pd.read_csv(db_path)

# Grab the first 600 unique patient IDs from the dataset
unique_patients = df['patient_id'].unique()[:600]
print(f"[Info] Found the first 600 patient IDs to download: {list(unique_patients)}")

# Filter the dataframe to only contain these 600 patients
filtered_df = df[df['patient_id'].isin(unique_patients)]
print(f"[Dataset] Found {len(filtered_df)} corresponding data rows. Starting direct download...")

# Loop through each row to download its respective files
for idx, row in filtered_df.iterrows():
    patient = row['patient_id']
    step = row['step']
    
    print(f"\n[Processing] Fetching files for Patient {patient} ({step} exercise step)...")
    
    # List of files specified in your CSV row format
    files_to_download = [
        row['video'],     # Video file pathway
        row['ppg'],       # PPG array pathway
        row['ecg'],       # ECG array pathway
        row['meta'],      # Text meta pathway
        row['ppg_sync']   # Synced PPG pathway
    ]
    
    for file_path in files_to_download:
        if pd.isna(file_path):
            continue
            
        print(f"   -> Downloading: {file_path}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=output_dir
            )
        except Exception as e:
            print(f"   ! Skipped/Failed to download {file_path}: {e}")

print(f"\n[Success] Clean download complete! Everything is saved to: {os.path.abspath(output_dir)}")