# SemEval 2025 Task 10 — Narrative Classification (Team 16 “Apfelkuchen”)

University team project for [SemEval-2025 Task 10](https://propaganda.math.unipd.it/semeval2025task10/): *Multilingual Characterization and Extraction of Narratives from Online News*. We worked on **Subtask 2 — Narrative Classification**.

**Team:** Aleksandr Pavlenko, Vadim Chaikin, Nikita Morev, Fedor Gerasimov, Farkhat Almukhamedov

📄 **Poster with full results:** [Team_16.pdf](Team_16.pdf)

## Task

Given a news article and a two-level taxonomy of labels (narratives → subnarratives), assign all applicable subnarrative labels to the article. This is a **hierarchical multi-label, multi-class document classification** problem.

The data consists of news articles on the **Russia–Ukraine war** and **climate change** in **five languages**: English, Portuguese, Hindi, Bulgarian, and Russian.

## What we did

**1. Data analysis and preprocessing** ([`dataset_analyze.py`](dataset_analyze.py))
- Corpus statistics (word counts, vocabulary size, sentence length), POS-tag distribution, word frequencies, and word clouds (see [`plots/`](plots/))
- Preprocessing: removal of punctuation, numbers and language-specific stopwords; lemmatization and stemming
- Identified data issues: duplicate texts under different IDs, and inconsistent content in labels marked “Other”

**2. Models** — three approaches built on `bert-base-multilingual-uncased`:

| Approach | Idea |
|---|---|
| **Base** | BERT classifier trained on the sum of narrative and subnarrative losses |
| **Masked** | Masked loss that only allows subnarratives valid for the predicted narrative, enforcing the label hierarchy |
| **Tree** | BERT + graph convolution layers (GCNConv) that model the relations between label levels |

## Results

Incorporating the label hierarchy (Masked and Tree models) clearly outperformed the flat Base model; the GCN-based model performed best during training. Proper preprocessing gave a small additional improvement.

Test-set scores of our submission:

| Language | F1 macro (coarse) | F1 samples |
|---|---|---|
| English | 0.392 | 0.336 |
| Portuguese | 0.670 | 0.434 |
| Hindi | 0.345 | 0.143 |
| Bulgarian | 0.526 | 0.300 |
| Russian | 0.516 | 0.248 |

## Repository structure

```
├── dataset_analyze.py                    # exploratory data analysis & plots
├── plots/                                # word frequencies, word clouds, POS distribution
├── data/                                 # official training and dev data
├── semeval2025task10-scorers-baselines/  # official scorers and baselines
├── Team_16.pdf                           # project poster
└── requirements.txt
```

## Setup

Python 3.12.7

```bash
pip install -r requirements.txt
python dataset_analyze.py
```

## Tech stack

Python · PyTorch · Hugging Face Transformers (mBERT) · graph convolutional networks · NLTK · spaCy · scikit-learn · pandas · matplotlib

## References

1. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805), 2018.
2. Wang et al. *Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification*. [arXiv:2203.03825](https://arxiv.org/abs/2203.03825), 2022.
3. Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907), 2016.
