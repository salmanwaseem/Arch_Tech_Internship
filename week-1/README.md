# Week 1 - Email Spam Dataset Cleaning

## Overview

This project is part of the **Arch Technologies Internship**.

The objective of Week 1 is to prepare and clean a labelled email dataset that can later be used to train a machine learning model for spam email classification.

The dataset contains raw email text along with a binary label indicating whether an email is spam or legitimate.

## Technologies Used

- Python
- Pandas
- Regular Expressions
- Git
- GitHub

## Dataset

The dataset used for this project was obtained from Kaggle:

[Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

The dataset contains labelled email messages that can be used for spam classification.

### Dataset Columns

The CSV file contains two main columns:

| Column | Description |
|---|---|
| `label` | Binary classification label for the email |
| `text` | Raw email message content |

### Label Mapping

- `0` = Ham / legitimate email
- `1` = Spam email

### Dataset Statistics

Before preprocessing, the dataset contained:

- Total rows: **83,448**
- Total columns: **2**
- Ham emails: **39,538**
- Spam emails: **43,910**
- Missing label values: **0**
- Missing text values: **0**
- Unique labels: `0` and `1`

### Dataset Availability

The dataset files are not included in this GitHub repository because of their large file size.

To reproduce the preprocessing:

1. Download the dataset from Kaggle using the link above.
2. Place the downloaded CSV file inside:

```text
week-1/data/
```

3. Rename the downloaded dataset to:

```text
spam_raw.csv
```

The expected file path is:

```text
week-1/data/spam_raw.csv
```

After running the preprocessing script, the cleaned dataset will automatically be generated as:

```text
week-1/data/spam_cleaned.csv
```

Both CSV files are excluded from Git tracking using the repository's `.gitignore` file.

## Data Cleaning

The preprocessing script performs the following operations:

1. Loads the raw email dataset using Pandas.
2. Keeps only the required `label` and `text` columns.
3. Removes missing values.
4. Removes rows containing invalid labels.
5. Removes leading and trailing whitespace.
6. Removes empty emails.
7. Removes exact duplicate emails.
8. Converts email text to lowercase.
9. Removes HTML tags.
10. Replaces URLs with a `url` token.
11. Replaces email addresses with an `emailaddress` token.
12. Replaces numbers with a `number` token.
13. Removes punctuation and special characters.
14. Normalizes extra whitespace.
15. Removes emails that become empty after preprocessing.
16. Removes duplicate emails created after text normalization.
17. Saves the cleaned dataset as a new CSV file.

## Cleaning Results

| Metric | Count |
|---|---:|
| Original rows | 83,448 |
| Exact duplicates removed | 75 |
| Empty rows after cleaning | 13 |
| Normalized duplicates removed | 1,711 |
| Final rows | 81,649 |

### Final Label Distribution

| Label | Type | Count |
|---|---|---:|
| `0` | Ham | 38,009 |
| `1` | Spam | 43,640 |

The final cleaned dataset contains no missing values in the `label`, `text`, or `clean_text` columns.

## Output Dataset

The cleaned dataset contains three columns:

| Column | Description |
|---|---|
| `label` | Email classification label |
| `text` | Original email text |
| `clean_text` | Cleaned and normalized email text |

## Project Structure

```text
Arch_Tech_Internship/
│
├── .gitignore
│
└── week-1/
    │
    ├── data/
    │   ├── spam_raw.csv       # Local only - ignored by Git
    │   └── spam_cleaned.csv   # Generated locally - ignored by Git
    │
    ├
    │
    ├── src/
    │   └── preprocess.py
    │
    ├── README.md
    └── requirements.txt
```

## Installation

Create a Python virtual environment from the root directory of the repository:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install -r week-1\requirements.txt
```

## Running the Preprocessing Script

From the root directory of the repository, run:

```bash
python week-1\src\preprocess.py
```

The script will:

- Load the raw dataset
- Clean the email text
- Remove duplicate and invalid records
- Display a cleaning summary
- Save the cleaned dataset to:

```text
week-1/data/spam_cleaned.csv
```
