from huggingface_hub import hf_hub_download
import os

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

# List of model filenames
model_files = [
    "MedSAM2_2411.pt",
    "MedSAM2_US_Heart.pt",
    "MedSAM2_MRI_LiverLesion.pt",
    "MedSAM2_CTLesion.pt",
    "MedSAM2_latest.pt"
]

# Download all models
for model_file in model_files:
    local_path = os.path.join("checkpoints", model_file)
    hf_hub_download(
        repo_id="wanglab/MedSAM2",
        filename=model_file,
        local_dir="checkpoints",
        local_dir_use_symlinks=False
    )
    print(f"Downloaded {model_file} to {local_path}")
