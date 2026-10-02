# Week 3: MNIST Digit Recognition

This week begins Task 2 of the internship project: recognizing handwritten digits from the MNIST dataset. The work here is intentionally limited to dataset acquisition, exploration, visualization, and image pre-processing. Model training and evaluation are planned for next week.

## Dataset

Download the MNIST dataset from Kaggle and place the files in `data/`. The notebook supports:

- Kaggle CSV files such as `mnist_train.csv`, with the digit label in the first column and 784 pixel columns after it.
- NumPy arrays such as `images.npy` and `labels.npy`.
- A single `.npz` file containing arrays named `images` and `labels` (or `x` and `y`).

One commonly used Kaggle source is [MNIST Handwritten Digit Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). Do not commit a large downloaded dataset unless your internship repository requires it. The included `.gitkeep` file keeps the empty data directory visible in Git.

## What is implemented

- Flexible loading of CSV and NumPy image data.
- Dataset shape and sample-count checks.
- Missing-value and basic corrupted-entry checks.
- A sample-image grid covering digit classes 0 through 9.
- A class-distribution bar chart.
- Pixel normalization to the 0-1 range.
- Image reshaping to flattened vectors or image tensors.
- An 80/20 train-test split using scikit-learn.
- Saving processed arrays as a reusable `.npz` file.

## Running the notebook

From this folder, install the dependencies and start Jupyter:

```text
pip install -r requirements.txt
jupyter notebook notebooks/mnist_eda.ipynb
```

If no dataset has been downloaded yet, the notebook explains the expected file names and continues with a small synthetic demo so that its visualizations can still be inspected. Replace the demo data with the Kaggle files before using the saved arrays for modeling.

## Next week

The next stage will train and evaluate digit-classification models using the processed arrays created here.
