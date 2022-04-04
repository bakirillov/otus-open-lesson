# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
# ---

# %% tags=["soorgeon-imports"]
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import pandas as pd

# %% tags=["parameters"]
upstream = None
product = None

# %% [markdown]
# ## Merge

# %%
np.random.seed(134)

# %%
calories = pd.read_csv("../data/calories.csv")
excersize = pd.read_csv("../data/exercise.csv")

# %%
df = pd.merge(calories, excersize)

# %% tags=["soorgeon-pickle"]
Path(product['df']).parent.mkdir(exist_ok=True, parents=True)
df.to_csv(product['df'], index=False)
