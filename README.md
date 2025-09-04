# Twitter Emotion Classification Challenge

This repository contains my solution to the **Natural Language Processing Assignment – Emotions in Tweets**.  
The task was to build a text classification model that can identify the primary emotion expressed in a tweet, and to package it as a pip-installable Python library with a command-line interface.

Kaggle Competition: [Link](https://www.kaggle.com/t/ded85d1670ed443a810d5d6c32747487)  
GitHub Repository: [ml_emotions](https://github.com/pilarguerreromorales/ml_emotions)

---

## Installation

The package can be installed directly via pip:

```bash
pip install git+https://github.com/pilarguerreromorales/ml_emotions.git
```

---

## Command Line Interface (CLI)

The repository provides a CLI tool called `inference` with the following commands:

### Predict emotion of a text
```bash
inference --input "This is a stupid idea"
```
Output:
```
anger
```

### Show Kaggle ID
```bash
inference --kaggle
```
Output:
```
your_kaggle_username
```

---

## Approach and Methodology

- **Data Exploration**: initial analysis revealed a slight class imbalance (labels 0 and 1 were overrepresented).  
- **Preprocessing**: normalized Unicode, lowercased text, removed URLs, mentions, punctuation, hashtags, and outliers (very short or long tweets).  
- **Class Imbalance Handling**: computed inverse-frequency class weights and used a `WeightedRandomSampler` to ensure fair sampling.  
- **Model Choice**:  
  - Attempted **RoBERTa**, but limited GPU memory prevented training.  
  - Successfully fine-tuned **DistilRoBERTa**, which proved effective and efficient.  
- **Training Setup**:  
  - Classification head randomly initialized.  
  - Fine-tuned for 4 epochs.  
  - Early stopping triggered at epoch 4.  
- **Optimization**: AdamW optimizer, linear warm-up followed by cosine decay, base LR = `2e-5`, gradient clipping (`norm=1.0`), dropout for regularization.  
- **Performance**: Achieved **Val F1 = 0.9176** at epoch 3.

---

## 📊 Experimental Results

| Trial | Description | Train Loss | Val F1 |
|-------|-------------|------------|--------|
| 1 | Full RoBERTa (resource-limited, failed to converge) | – | – |
| 2 | DistilRoBERTa (no regularization, CE loss) | 0.2578 | 0.914 |
| 3 | DistilRoBERTa + regularization, optimizers, schedulers | 0.1005 | **0.9176** |

---

## 📈 Training Curves

Here is the training vs validation loss curve for the final model:

![Train vs Validation Loss](Screenshot%202025-09-04%20at%2010.46.26.png)

---

## Assignment Deliverables

- Pip-installable package   
- CLI tool (`inference`)   
- Report (see attached PDF)   
