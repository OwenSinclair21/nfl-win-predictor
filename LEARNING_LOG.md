# Learning Log

## Session 1 - Project Setup

### What I did

- Created a dedicated Conda environment called `nfl-ml`
- Created the project folder structure
- Connected the Conda environment to JupyterLab
- Created the first notebook: `01_data_exploration.ipynb`

### Concepts learned

- A Conda environment isolates the Python packages used by this project.
- Jupyter notebooks allow code, outputs, graphs, and explanations to exist together.
- The project will be built incrementally rather than starting with a finished model.

### Next Step

Import the Python libraries and begin exploring NFL data.

## Session 2 - Loading and Inspecting NFL Data

### Concepts learned

- A DataFrame is a table consisting of rows and columns.
- In this dataset, each row represents an NFL game.
- Columns represent variables describing each game.
- `nflreadpy` returns a Polars DataFrame, which was converted to Pandas for this project.
- `head()`, `shape`, `columns`, `info()`, and `sample()` are basic tools for exploring an unfamiliar dataset.

## Session 3 - Building the Game Backbone

### What I did

- Filtered the schedule dataset to regular-season games.
- Selected the columns needed to identify each game.
- Checked the dataset for missing scores and tied games.
- Removed tied games for the initial binary classification problem.
- Created the target variable `home_win`.
- Examined the class distribution to establish a future baseline.
- Saved raw and processed versions of the dataset.

### Concepts learned

- Boolean filtering allows rows to be selected based on conditions.
- The target is the value the machine-learning model will attempt to predict.
- Features must only contain information available before the prediction event.
- Postgame information can be used to construct the target but cannot be used as model input.
- A baseline provides a simple benchmark that a machine-learning model should outperform.
- Raw data should remain unchanged while transformed data is stored separately.

## Session 4 - Creating Pregame Team Features

### What I did

- Converted game-level data into team-level observations.
- Represented each NFL game from both teams' perspectives.
- Calculated points for, points against, and point differential.
- Created rolling five-game pregame statistics.
- Used `shift(1)` to ensure the current game is excluded from its own features.

### Concepts learned

- Feature engineering transforms raw data into information useful for a model.
- Rolling windows summarize recent performance.
- `shift(1)` prevents current-game information from leaking into pregame features.
- Each feature must represent information that would have been known before kickoff.
- Early-season games create missing-data problems because little current-season history exists.

## Session 5 - Offensive and Defensive Efficiency

### What I did

- Created efficiency metrics including passing EPA per dropback, rushing EPA per carry, sack rate, and turnovers.
- Derived defensive performance from opponent offensive statistics.
- Joined offensive and defensive information into one team-level dataset.
- Created rolling five-game pregame efficiency metrics.

### Concepts learned

- Efficiency metrics can be more informative than raw volume statistics.
- EPA measures how much a play changes expected scoring value.
- Opponent offensive performance can be used to measure defensive performance.
- A self-join can combine related observations from the same dataset.
- Rolling features must use `shift(1)` to prevent information from the current game from leaking into predictions.

## Session 6 - Building the Game-Level Dataset

### What I did

- Merged each team's rolling pregame statistics onto the game-level schedule.
- Created separate home-team and away-team feature columns.
- Standardized feature naming across the dataset.
- Verified that the game-level merge preserved one row per NFL game.
- Began checking missing values before constructing the final ML dataset.

### Concepts learned

- `isna()` identifies missing values in a Pandas DataFrame.
- `sum()` can count missing values because boolean `True` values behave like 1.
- `any(axis=1)` checks whether any column in a row satisfies a condition.
- `dropna(subset=...)` removes rows containing missing values in specified columns.
- Missing data should be investigated before being removed.
- Week 1 naturally has missing rolling statistics because there are no earlier games in the same season.
- A model-training dataset should be validated before training begins.

### Current Dataset

The current game-level dataset contains:

- One row per NFL game
- Home-team pregame offensive and defensive statistics
- Away-team pregame offensive and defensive statistics
- Rest information
- Binary target: `home_win`

### Next Step

Investigate missing feature values, finalize Dataset V1, and then split the data chronologically into training, validation, and test sets.

### Missing Data Investigation

The first cleaned dataset dropped 290 games because one or more model features were missing.

I learned that missing values should not be removed blindly.

The missing-value analysis showed that all home-team features were missing together in many rows, and all away-team features were also missing together. This suggested a merge-key problem rather than individual feature calculation errors.

I also learned:

- `.any(axis=1)` checks whether a row contains at least one matching condition.
- `pd.concat()` can combine multiple Series into one longer Series.
- `.value_counts()` counts how often each distinct value occurs.
- `.unique()` returns the distinct values in a column.
- Systematic missing data can create selection bias if rows are dropped without investigation.

The next step is to verify whether historical NFL team abbreviations are causing join failures between the schedule and team-stat datasets.

### Historical Team Identifier Bug

The missing-data analysis revealed that entire blocks of home or away features were missing together.

The cause was inconsistent NFL team abbreviations between two data sources.

Historical schedule data used:

- OAK - Oakland Raiders
- SD - San Diego Chargers
- STL - St. Louis Rams

The team statistics dataset used modern canonical abbreviations:

- LV - Las Vegas Raiders
- LAC - Los Angeles Chargers
- LA - Los Angeles Rams

Because Pandas joins require exact matches, values such as `OAK` and `LV` were treated as completely different teams. This caused the team profile merge to fail and produced NaN values for the entire team's feature set.

I learned:

- Merge keys must use consistent identifiers across datasets.
- Categorical values should be standardized before downstream processing.
- `.replace(mapping_dictionary)` can standardize categorical values.
- Fixing data inconsistencies upstream is better than patching the final dataset.
- Missing data can expose problems in earlier stages of a data pipeline.
- Game IDs should remain unchanged because they are identifiers rather than team-name categories.

This reinforced why missing values should be investigated rather than immediately dropped.

### Missing Data Resolution

After standardizing historical team abbreviations across the schedule and team-stat datasets, the number of games removed because of missing model features dropped from 290 to 174.

This recovered 116 games that had previously been lost because team identifiers did not match during the merge.

The remaining missing rows were verified to correspond to teams playing their first game of the season. This includes an unusual case where Tampa Bay and Miami played their first game in Week 2.

This showed me that:

- Missing data should be investigated before being removed.
- Merge-key inconsistencies can silently remove large amounts of usable data.
- The first game of a season has no current-season rolling history, regardless of its official week number.
- Validation checks are important before model training begins.

## Session 7 - Preparing Data for Modeling

### Dataset Split

The final Version 1 dataset contains 2,711 usable games and 17 pregame features.

The dataset was split chronologically:

- Training: 2,200 games from 2015–2023
- Validation: 256 games from 2024
- Test: 255 games from 2025

### Concepts Learned

- `X` represents the model's input features.
- `y` represents the target the model is trying to predict.
- A Boolean mask can select rows that satisfy a condition.
- `.loc[mask]` selects rows where the mask is `True`.
- Sports prediction data should be split chronologically rather than randomly because the goal is to predict future games using past information.
- The training set is used to fit the model.
- The validation set is used to compare modeling decisions.
- The test set should remain untouched until final evaluation.
- A baseline provides a simple benchmark that a useful model should outperform.

### Development Environment Issue

The first attempt to train logistic regression caused the Jupyter kernel to crash even on a small synthetic dataset.

The model worked successfully when run directly from an activated Conda terminal, which showed that the data and scikit-learn code were not the cause.

The issue was isolated to how VS Code was launching the Conda Jupyter kernel. After correcting the VS Code environment activation behavior, the logistic regression pipeline successfully trained inside the notebook.

This taught me that a Python environment includes more than the Python executable itself. Environment variables, DLL paths, and how a process is launched can affect compiled numerical libraries used by packages such as NumPy, SciPy, and scikit-learn.

## Session 8 - First Machine Learning Model

### Model

I trained the first version of the NFL prediction model using logistic regression.

The model uses a scikit-learn Pipeline containing:

1. StandardScaler
2. LogisticRegression

The scaler learns feature means and standard deviations only from the training set before the classifier is trained.

### Validation Results

- Home-team baseline accuracy: 52.34%
- Logistic regression validation accuracy: 64.06%
- Improvement over baseline: 11.72 percentage points
- Validation log loss: 0.6485

The model was trained using games from 2015-2023 and evaluated on the unseen 2024 season.

### Concepts Learned

- `.fit()` learns model parameters from training data.
- `.predict()` returns predicted classes.
- `.predict_proba()` returns probabilities for each class.
- Accuracy measures the percentage of correct classifications.
- Log loss evaluates probability quality and penalizes confident incorrect predictions.
- Logistic regression learns one coefficient for each input feature.
- Positive coefficients push predictions toward the positive class (`home_win = 1`).
- Negative coefficients push predictions toward the negative class.
- Standardization makes logistic-regression coefficients easier to compare because features are placed on similar scales.
- Model coefficients represent predictive relationships and should not automatically be interpreted as causal effects.