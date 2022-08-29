# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Sharing
#
# Some journals will require the sharing of code or models, but even if they don’t we might benefit from it.
#
# Anytime we share a model, we give other researchers the opportunity to replicate our studies and iterate upon them. Altruistically, this advances science, which in and of itself is a noble pursuit. However, this also increases the citations of our original research, a core metric for most researchers in academia.
#
# In this section, we explore how we can export models and make our training codes reproducible. Saving a model from scikit-learn is easy enough. But what tools can we use to easily make our training code adaptable for others to import and try out that model? Specifically, I want to talk about:
#
# - Automatic Linters
# - Automatic Formatting
# - Automatic Docstrings and Documentation
# - Docker and containerization for ultimate reproducibility
#

# %% [markdown]
# ## Model Export
# Scikit learn uses the Python `pickle` (or rather `joblib`) module to persist models in storage.
# More information [here](https://scikit-learn.org/stable/model_persistence.html)
# %%
import pandas as pd
penguins = pd.read_csv('../data/penguins_clean.csv')
penguins.head()

# %%
from sklearn.model_selection import train_test_split
num_features = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]
cat_features = ["Sex"]
features = num_features + cat_features
target = ["Species"]

X_train, X_test, y_train, y_test = train_test_split(penguins[features], penguins[target[0]], stratify=penguins[target[0]], train_size=.7, random_state=42)

# %%
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC()),
])

model.fit(X_train, y_train)
model.score(X_test, y_test)

# %%
from joblib import dump, load
dump(model, "../model/svc.joblib")
clf = load("../model/svc.joblib")
clf.score(X_test, y_test)

# %% [markdown]
# ## Sources of Randomness
# You may have noticed that I used `random_state` in some arguments.
#
# This fixes all sources of random initialization in a model or method to this particular random seed.
#
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Always Google the latest way to fix randomness in machine learning code. It differs from library to library and version to version.<br>It's easy to forget one, which defeats the entire purpose.</div>
#
# This works in making models reproducible. Try changing the random seed below!

# %%
from sklearn.model_selection import cross_val_score

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42)),
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
scores

# %% [markdown]
# ## Good code practices
# ### Linting
# Tools like linters are amazing at cleaning up code. `flake8` and editors like VSCode can find unused variables, trailing white-space and lines that are way too long.
#
# They immediately show some typos that you would otherwise have to pain-stakingly search.
#
# [Flake8](https://flake8.pycqa.org/en/latest/) tries to be as close to the PEP8 style-guide as possible and find bugs before even running the code.
#
# ### Formatters
# There are automatic formatters like `black` that will take your code and change the formatting to comply with formatting rules.
#
# Formatters like [black](https://github.com/psf/black) don't check the code itself for bugs, but they're great at presenting a common code style.
#
# They're my shortcut to make good-looking code and make collaboration 100 times easier as the formatting is done by an automated tool.
#
# ### Docstrings
# Python has documentation built into its core.
# For example, below is the SVC model, if you put the cursor in the brackets, you can hit `Shift + Tab` in Jupyter to read the documentation that is written in the docstring.

# %%
SVC()

# %% [markdown]
# In VSCode for example there are tools that [autogenerate a docstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) based on a function footprint:
#
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Docstrings are an essential part in telling users what a function does and what input and outputs are expected.</div>
# I see docstrings as the minimum documentation one should provide when sharing code, and the auto-generators make it extremely easy to do so.
#
# This docstrin was automatically generated. Just fill in the summary and description and people are happy!
# ```python
# def hello_world(name: str) -> None:
#     """_summary_
#
#     Parameters
#     ----------
#     name : str
#         _description_
#     """
#     print(f"Hello {name}"")
# ```

# %% [markdown]
# ## Dependencies
# This repository comes with a `requirements.txt` and a `environment.yml` for `pip` and `conda`.
#
# A `requirements.txt` can consist of simply the package names. But ideally you want to add the version number, so people can automatically install the exact version you used in your code.
# This looks like `pandas==1.0`
#
# The conda `environment.yml` can be auto-exported from your conda environment:
#
# `conda env export --from-history > environment.yml`
# The `--from-history` makes it cross-platform but eliminates the version numbers.

# %% [markdown]
# ## Docker for ultimate reproducibility
# Docker is container that can ship an entire operating system with installed packages and data.
#
# It makes the environment you used for your computation almost entirely reproducible.
#
# (It's also great practice for the business world)
#
# Docker containers are build using the `docker build` command using a `Dockerfile`, an example docker file for Python looks like this:
#
# ```docker
# # syntax=docker/dockerfile:1
#
# FROM python:3.8-slim-buster
#
# WORKDIR /
#
# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt
#
# COPY . .
#
# CMD python train.py
# ```
