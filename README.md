# Titanic Disaster: Survival Prediction (Python + R in Docker)

This repository predicts which passengers survived the sinking of the Titanic (April 15, 1912) using logistic regression. The same pipeline is implemented twice, once in **Python** and once in **R**, and each runs inside its own Docker container, so you don't need Python or R installed on your machine.

Each script loads the data, explores it, cleans it and engineers features, trains a logistic regression model, reports accuracy, and predicts on the test set. Every step prints to the terminal, so you can follow the whole run without opening the code.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS               # repo ownership / required reviewer
├── .gitignore
├── README.md
└── src/
    ├── data/                    # put the Kaggle CSVs here (ignored by git)
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

The dataset is **not** included in this repository. Step 2 below explains how to download it.

---

## Prerequisites

1. **Docker**. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or Docker Engine (Linux), and make sure it is **running** before you continue. Check with:
   ```bash
   docker --version
   ```
2. **Git**, to clone the repository.
3. **A free Kaggle account**, to download the data.

---

## Step 1: Clone the repository

```bash
git clone https://github.com/skedila/titanic-disaster.git
cd titanic-disaster
```

**All remaining commands are run from this repository root folder** (the folder containing this README).

---

## Step 2: Download the data

1. Go to the competition data page: <https://www.kaggle.com/competitions/titanic/data>
2. Sign in, and if prompted, click **"Join Competition"** / accept the rules (Kaggle requires this before downloading).
3. Click **"Download All"**. This downloads `titanic.zip`.
4. Unzip it and move these three files into `src/data/`:
   - `train.csv`
   - `test.csv`
   - `gender_submission.csv`

From the terminal (Mac/Linux), assuming the zip landed in your Downloads folder:

```bash
unzip ~/Downloads/titanic.zip -d src/data/
```

<details>
<summary>Alternative: download with the Kaggle CLI</summary>

If you have the Kaggle API set up (`pip install kaggle` and an API token in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data && rm src/data/titanic.zip
```
</details>

Verify the files are in place:

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

---

## Step 3: Run the Python model

```bash
# Build the image (first build takes ~1 minute)
docker build -t titanic-python src/python

# Run the container, mounting the data folder into it
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

## Step 4: Run the R model

```bash
# Build the image (first build takes a few minutes while R packages install)
docker build -t titanic-r src/r

# Run the container, mounting the data folder into it
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

> **Windows users:** in PowerShell replace `"$(pwd)/src/data"` with `"${PWD}/src/data"`. In Command Prompt use `"%cd%/src/data"`.

### Why the `-v` flag?

The data is not baked into the Docker images (it shouldn't be shipped with the code). Instead, `-v "$(pwd)/src/data:/app/data"` mounts your local `src/data` folder into the container at `/app/data`, where the scripts look for it. This is also how the prediction files get written back to your machine.

---

## What you'll see

Both scripts print the same six stages:

| Stage | What happens |
|---|---|
| 1. Loading | Loads `train.csv`, prints its shape and first rows |
| 2. Exploring | Overall survival rate, and survival rate by sex and passenger class |
| 3. Cleaning and features | Extracts title from name (Mr, Mrs, Miss, Master, Rare); fills missing Age with the median age for that title; fills missing Fare and Embarked; encodes Sex; adds `FamilySize`, `IsAlone` and `HasCabin`; one-hot encodes categories; drops Name, Ticket and Cabin. Each change prints how many values it affected. |
| 4. Training | Fits a logistic regression, prints hold-out validation accuracy (80/20 split), training accuracy, and the model coefficients |
| 5. Test predictions | Applies the same cleaning to `test.csv` (using values learned from the training data only), predicts survival, and saves predictions |
| 6. Test accuracy | See note below |

### Output files

After running, the predictions are saved to your local data folder in Kaggle submission format:

- `src/data/predictions_python.csv`
- `src/data/predictions_r.csv`

### A note on test accuracy

Kaggle's `test.csv` **does not contain the `Survived` column**; the true labels are held by Kaggle. So the scripts report:

- **Hold-out validation accuracy**: 20% of the training set is held out, the model is trained on the other 80%, and accuracy is measured on the held-out part. This is the best offline estimate of test accuracy.
- **Agreement with `gender_submission.csv`**: how often the predictions match Kaggle's example baseline ("all women survive, all men die"). This is a comparison with a baseline, not true accuracy.

To get the true test accuracy, submit either predictions file at <https://www.kaggle.com/competitions/titanic/submissions>.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: .../train.csv not found` | The data isn't in `src/data/`, or you ran `docker run` from a folder other than the repo root. `cd` to the repo root and check `ls src/data`. |
| `Cannot connect to the Docker daemon` | Docker Desktop isn't running. Start it and retry. |
| Permission error on the volume (Mac) | Docker Desktop → Settings → Resources → File sharing: make sure the folder containing the repo is shared. |

---

## Package versions

**Python** (`src/python/requirements.txt`, on `python:3.11-slim`): numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1

**R** (on `rocker/r-ver:4.4.1`): readr, dplyr. The rocker image pins CRAN to a dated snapshot for R 4.4.1, so the installed package versions are fixed and reproducible.
