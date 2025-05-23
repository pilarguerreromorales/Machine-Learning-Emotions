import click
from inference.model import get_model

LABELS    = ["sadness","joy","love","anger","fear","surprise"]
KAGGLE_ID = "pilarguerreromorales"  # your actual Kaggle username

@click.command()
@click.option("--input", "-i", "text", help="Text to classify")
@click.option("--kaggle", is_flag=True, help="Show Kaggle ID")
def main(text, kaggle):
    """
    Emotion classifier CLI.
    Usage:
      inference --input "Some text"
      inference --kaggle
    """
    if kaggle:
        click.echo(KAGGLE_ID)
        return

    if text:
        model, tokenizer = get_model()
        enc    = tokenizer(text, return_tensors="pt", truncation=True)
        logits = model(**enc).logits
        pred   = logits.argmax(dim=-1).item()
        click.echo(LABELS[pred])
        return

    click.echo(main.get_help(click.Context(main)))

if __name__ == "__main__":
    main()
