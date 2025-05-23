import click
import torch
from inference.model import get_model

LABELS    = ["sadness","joy","love","anger","fear","surprise"]
KAGGLE_ID = "pilarguerreromorales"  # replace with your actual Kaggle username

@click.group()
def main():
    """Emotion classifier CLI."""
    pass

@main.command("input")
@click.option("--input", "-i", "text", required=True, help="Text to classify")
def inference(text):
    """Predict emotion."""
    model, tokenizer = get_model()
    enc    = tokenizer(text, return_tensors="pt", truncation=True)
    logits = model(**enc).logits
    pred   = logits.argmax(dim=-1).item()
    click.echo(LABELS[pred])

@main.command("kaggle")
def kaggle():
    """Show Kaggle ID."""
    click.echo(KAGGLE_ID)

if __name__ == "__main__":
    main()
