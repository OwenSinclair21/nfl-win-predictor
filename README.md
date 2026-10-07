# NFL Win Prediction Model

A machine learning project focused on building, training, and evaluating models that predict the outcome of NFL games using information available before kickoff.

## Project Goal

The primary goal of this project is to learn the complete machine learning workflow, including:

- Data collection and exploration
- Data cleaning
- Feature engineering
- Avoiding data leakage
- Train/validation/test splitting
- Building baseline models
- Training classification models
- Model evaluation
- Hyperparameter tuning
- Error analysis
- Model improvement

## Current Status

Project setup and initial data exploration.

## Current Status

The initial game-level machine learning dataset is under construction.

Completed:

- Historical NFL schedule ingestion
- Regular-season game backbone
- Binary win/loss target
- Team-level offensive efficiency metrics
- Team-level defensive efficiency metrics
- Rolling five-game pregame features
- Home and away team feature joins
- Dataset validation and missing-value analysis

Next:

- Finalize the Version 1 feature set
- Create chronological train/validation/test splits
- Establish baseline prediction performance
- Train and evaluate the first logistic regression model

## Current Status

The Version 1 game-level dataset has been completed.

The current pipeline:

1. Loads historical NFL regular-season games from 2015–2025.
2. Standardizes historical franchise identifiers across datasets.
3. Creates team-level offensive and defensive efficiency metrics.
4. Calculates rolling pregame features using only prior games.
5. Merges home and away team profiles onto each game.
6. Validates missing data and merge consistency.
7. Removes games without sufficient prior-season history.
8. Splits the dataset chronologically:
   - 2015–2023: training
   - 2024: validation
   - 2025: test

The Version 1 model uses 17 pregame features across 2,711 games.

Next steps:

- Establish baseline performance
- Train a logistic regression classifier
- Evaluate validation performance
- Analyze feature coefficients
- Experiment with matchup-specific features