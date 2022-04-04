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
import sweetviz as sw
from pathlib import Path
import pickle
import pandas as pd

# %% tags=["parameters"]
upstream = ['merge']
product = None

# %% tags=["soorgeon-unpickle"]
df = pd.read_csv(upstream['merge']['df'])

# %% [markdown]
# ## EDA

# %%
my_report = sw.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"
