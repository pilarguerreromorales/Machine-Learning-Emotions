
import gdown
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME    = "distilroberta-base"
MODEL_PATH    = Path(__file__).parent.parent / "models" / "emotion_model.pt"


def download_from_cloud_storage(dest: Path):
    """
    Download a large file from Google Drive using gdown,
    which handles confirmation tokens automatically.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/file/d/1qEy4bXsFFx338ay0kRvAH5qr7PN2xpYJ/view?usp=sharing"
    # will overwrite if exists; quiet=False shows progress
    gdown.download(url, str(dest), quiet=False, fuzzy=True)

def get_model():
    # Download once
    if not MODEL_PATH.exists():
        print("Downloading model (this will only happen once)...")
        download_from_cloud_storage(MODEL_PATH)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6
    )
    # force full unpickle (weights_only=False) to load your fine-tuned state dict
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer
