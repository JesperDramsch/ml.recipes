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
#     display_name: Python 3.10.8 ('pydata-global-2022-ml-repro')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Benchmarking
#
# Another common reason for rejections of machine learning papers in applied science is the lack of proper benchmarks. This section will be fairly short, as it differs from discipline to discipline.
#
# However, any time we apply a superfancy deep neural network, we need to supply a benchmark to compare the relative performance of our model to. These models should be established methods in the field and simpler machine learning methods like a linear model, support-vector machine or a random forest.

# %%
from pathlib import Path

DATA_FOLDER = Path("..") / "data"
DATA_FILEPATH = DATA_FOLDER / "penguins_clean.csv"

# %%
import pandas as pd
penguins = pd.read_csv(DATA_FILEPATH)
penguins.head()

# %%
from sklearn.model_selection import train_test_split
num_features = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]
cat_features = ["Sex"]
features = num_features + cat_features
target = ["Species"]

X_train, X_test, y_train, y_test = train_test_split(penguins[features], penguins[target], stratify=penguins[target[0]], train_size=.7, random_state=42)
X_train

# %% [markdown]
# ## Dummy Classifiers
# One of the easiest way to build a benchmark is ensuring that our model performs better than random.
#
# <div class="alert alert-block alert-info">
# <b>Tip:</b> If our model is effectively as good as a coin flip, it's a bad model.</div>
# However, sometimes it isn't obvious how good or bad a model is. Take our penguin data. What counts as "random classification" on 3 classes that aren't equally distributed?

# %%
y_train.reset_index().groupby(["Species"]).count()

# %% [markdown]
# We can use the `DummyClassifier` and `DummyRegressor` to show what a random model would predict.

# %%
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %% [markdown]
# ## Benchmark Datasets
# Another great tool to use is benchmark datasets.
#
# Most fields have started creating benchmark datasets to test new methods in a controlled environment.
# Unfortunately, it still happens that results are over-reported because models weren't adequately evaluated as seen in [notebook 1](1 - Model Evaluation.ipynb).
# Nevertheless, it's easy to reproduce the results as both the code and data are available, so we can quickly see how legitimate reported scores are.
#
# Examples:
# - [Imagenet](https://www.image-net.org/) in computer vision
# - [WeatherBench](https://github.com/pangeo-data/WeatherBench) in meteorology
# - [ChestX-ray8](https://paperswithcode.com/dataset/chestx-ray8) in medical imaging

# %% [markdown]
# ## Domain Methods
# Any method is stronger if it is verified against standard methods in the field.
#
# A weather forecast post-processing method should be evaluated against a standard for forecast post-processing.
#
# This is where domain expertise is important.

# %% [markdown]
# ## Linear and Standard Models
# In addition to the Dummy methods, we also want to evaluate our fancy solutions against very simple models.
#
# Personally, I like using:
# - [Linear Models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# - [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
#
# As an exercise try implementing baseline models to compare against the support-vector machine with preprocessing.
