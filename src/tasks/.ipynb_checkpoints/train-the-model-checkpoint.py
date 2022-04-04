# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=["soorgeon-imports"]
import numpy as np
from ludwig import logging
from ludwig.api import LudwigModel
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import pandas as pd

# %% tags=["parameters"]
upstream = ['merge']
product = None

# %% tags=["injected-parameters"]
# This cell was injected automatically based on your stated upstream dependencies (cell above) and pipeline.yaml preferences. It is temporary and will be removed when you save this notebook
upstream = {"merge": {"df": "/home/bakirillov/Works/Otus/OpenLesson/src/output/merge-df.csv", "nb": "/home/bakirillov/Works/Otus/OpenLesson/src/output/merge.ipynb"}}
product = {"model": "/home/bakirillov/Works/Otus/OpenLesson/src/output/train-the-model-model.pkl", "test_df": "/home/bakirillov/Works/Otus/OpenLesson/src/output/train-the-model-test_df.pkl", "nb": "/home/bakirillov/Works/Otus/OpenLesson/src/output/train-the-model.ipynb"}


# %% tags=["soorgeon-unpickle"]
df = pd.read_csv(upstream['merge']['df'])

# %% [markdown] tags=[]
# ## Train the model

# %% tags=[]
df.columns

# %% tags=[]
input_features = [
    {"name": "Gender", "type": "category"},
    {"name": "Duration", "type": "numerical"},
    {"name": "Age", "type": "numerical"},
    {"name": "Height", "type": "numerical"},
    {"name": "Weight", "type": "numerical"},
    {"name": "Heart_Rate", "type": "numerical"},
    {"name": "Body_Temp", "type": "numerical"}
]

# %% tags=[]
# Define the model using input features and additional parameters
model_definition = {
    "input_features": input_features,
    "output_features": [{
        "name": "Calories", "type": "numerical"
    }],
    "preprocessing": {
        "numerical": {
            "normalization": "minmax"
        }
    },
    "training": {
        "batch_size": 128,
        "decay": True,
        "decay_rate": 0.96,
        "decay_steps": 10000,
        "early_stop": 10,
        "epochs": 100,
    }
}

# %% tags=[]
model = LudwigModel(model_definition, logging_level=logging.DEBUG)

# %% tags=[]
train_ix, test_ix = train_test_split(np.arange(df.shape[0]))

# %% tags=[]
train_df = df.iloc[train_ix]
test_df = df.iloc[test_ix]

# %% tags=[]
train_stats = model.train(train_df)

# %% tags=[]
model.save("trained")

# %% tags=["soorgeon-pickle"]
Path(product['model']).parent.mkdir(exist_ok=True, parents=True)
Path(product['model']).write_bytes(pickle.dumps(model))

Path(product['test_df']).parent.mkdir(exist_ok=True, parents=True)
Path(product['test_df']).write_bytes(pickle.dumps(test_df))
