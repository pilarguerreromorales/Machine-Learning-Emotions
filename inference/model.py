import requests
from pathlib import Path

# the same MODEL_NAME and MODEL_PATH you already have
MODEL_NAME = "distilroberta-base"
MODEL_PATH = Path(__file__).parent.parent / "models" / "emotion_model.pt"

# Your Google Drive file ID
GDRIVE_FILE_ID = "1AbCdEFGhIJklmNoPQRsTuvWXyz"

def download_from_cloud_storage(dest: Path):
    """
    Download the Google-Drive-hosted model to `dest`.
    """
    url = (
        "https://docs.google.com/uc?export=download"
        f"&id={GDRIVE_FILE_ID}"
    )
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1_024*1_024):
                if chunk:
                    f.write(chunk)

def get_model():
    # Only download once
    if not MODEL_PATH.exists():
        print("Downloading model (this will only happen once)...")
        download_from_cloud_storage(MODEL_PATH)

    # Load tokenizer & model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer

