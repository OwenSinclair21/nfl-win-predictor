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