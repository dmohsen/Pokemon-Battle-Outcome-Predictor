# Pokémon Battle Outcome Predictor

A machine learning project that predicts Pokémon battle outcomes using combat stats and attribute data. Built in Python using Scikit-learn, Pandas, and NumPy.

## Overview

Given two Pokémon and their base attributes (HP, Attack, Defense, Special Attack, Special Defense, Speed), this model predicts which one wins the battle — and identifies which attributes matter most.

**Key finding:** Speed is by far the dominant predictor of battle success, accounting for ~90% of feature importance in the final model. Faster Pokémon strike first, creating a compounding advantage that outweighs most other attributes.

## Dataset

- `pokemon.csv` — Base stats for 800 Pokémon
- `combats.csv` — 150,000+ battle records with outcomes
- `tests.csv` — Held-out test set for final evaluation

Data sourced from [Kaggle](https://www.kaggle.com/).

## Approach

### Data Preparation
- Merged battle records with Pokémon stats for both combatants
- Engineered win percentage feature from combat history (win count / total battles per Pokémon)
- Validated data integrity and enforced clean merge logic throughout

### Models Trained
| Model | Training Score | MAE |
|---|---|---|
| Linear Regression | 0.9008 | 0.0566 |
| Decision Tree | 0.9780 | 0.0490 |
| Random Forest | 0.9733 | 0.0444 |
| Neural Network | — | 0.1095 |

Random Forest achieved the lowest Mean Absolute Error and was selected for final predictions.

### Feature Importance
Top predictors from the Random Forest model:
1. **Speed** (~90% importance)
2. Attack
3. HP
4. Defense
5. Special Attack
6. Special Defense

A separate model trained on Speed alone confirmed its dominance — performance remained competitive even without other features.

## Results

- Final Random Forest model predicted battle winners across 10,000 test matchups
- Speed-only model vs. full model differed on only 602 out of 10,000 predictions (6%)
- Simpler models outperformed the Neural Network on this structured, low-dimensional dataset

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Random Forest, Decision Tree, Linear Regression)
- Keras / TensorFlow (Neural Network)
- Matplotlib / Seaborn (visualizations)
- Jupyter Notebook

## Files

```
├── pokemonproject.ipynb   # Full analysis notebook
├── pokemon.csv            # Pokémon base stats
├── combats.csv            # Battle records
├── tests.csv              # Test set
└── README.md
```

## Usage

```bash
git clone https://github.com/dmohsen/Pokemon-Battle-Outcome-Predictor
cd Pokemon-Battle-Outcome-Predictor
jupyter notebook pokemonproject.ipynb
```
