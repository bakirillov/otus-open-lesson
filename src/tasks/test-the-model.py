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
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import pandas as pd
from ludwig.api import LudwigModel

# %% tags=["parameters"]
upstream = ['train-the-model']
product = None

# %% tags=["soorgeon-unpickle"]
model = LudwigModel.load("trained")#pickle.loads(Path(upstream['train-the-model']['model']).read_bytes())
test_df = pickle.loads(Path(upstream['train-the-model']['test_df']).read_bytes())

# %% [markdown]
# ## Test the model

# %%
test_Yhat = model.predict(test_df)[0]

# %%
spearmanr(test_df["Calories"].values, test_Yhat["Calories_predictions"].values)

# %%
plt.scatter(test_df["Calories"].values, test_Yhat["Calories_predictions"].values)

# %%
mean_absolute_error(y_pred=test_Yhat["Calories_predictions"].values, y_true=test_df["Calories"].values)

# %%
model.save('trained')
