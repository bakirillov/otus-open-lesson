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
import pandas as pd
from ludwig.api import LudwigModel
from pathlib import Path
import pickle
import pandas as pd

# %% tags=["parameters"]
upstream = None
product = None

# %% [markdown]
# ## Predict

# %%
model = LudwigModel.load("trained")

# %%
d = pd.read_csv("test.csv", index_col=0)

# %%
Yhat = model.predict(d)[0]

# %%
Yhat.to_csv("predicted.csv")

# %% tags=["soorgeon-pickle"]
Path(product['model']).parent.mkdir(exist_ok=True, parents=True)
Path(product['model']).write_bytes(pickle.dumps(model))
