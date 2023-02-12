# ---
# jupyter:
#   jupytext:
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
# # Testing
#
# Machine learning is very hard to test. Due to the nature of the our models, we often have soft failures in the model that are difficult to test against.
#
# Writing software tests in science, is already incredibly hard, so in this section we’ll touch on 
#
# - some fairly simple tests we can implement to ensure consistency of our input data
# - avoid bad bugs in data loading procedures
# - some strategies to probe our models
#

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

# %%
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from joblib import load

MODEL_FOLDER = Path("..") / "model"
MODEL_EXPORT_FILE = MODEL_FOLDER / "svc.joblib"

clf = load(MODEL_EXPORT_FILE)
clf.score(X_test, y_test)

# %% [markdown]
# ## Deterministic Tests
# When I work with neural networks, implementing a new layer, method, or fancy thing, I try to write a test for that layer. The `Conv2D` layer in Keras and Pytorch for example should always do the same exact thing, when they convole a kernel with an image.
#
# Consider writing a small `pytest` test that takes a simple numpy array and tests against a known output.
#
# You can check out the `keras` test suite [here](https://github.com/keras-team/keras/tree/master/keras/tests) and an example how they validate the [input and output shapes](https://github.com/keras-team/keras/blob/18248b084f932e294402f0b772b49ed162c25208/keras/testing_infra/test_utils.py#L217).
#
# Admittedly this isn't always easy to do and can go beyond the need for research scripts.

# %% [markdown]
# ## Data Tests for Models
#
# An even easier test is by essentially reusing the notebook from the Model Evaluation and writing a test function for it.
#

# %%
def test_penguins(clf):
    # Define data you definitely know the answer to
    test_data = pd.DataFrame([[34.6, 21.1, 198.0, "MALE"],
                              [46.1, 18.2, 178.0, "FEMALE"],
                              [52.5, 15.6, 221.0, "MALE"]], 
             columns=["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Sex"])
    # Define target to the data
    test_target = ['Adelie Penguin (Pygoscelis adeliae)',
                   'Chinstrap penguin (Pygoscelis antarctica)',
                   'Gentoo penguin (Pygoscelis papua)']
    # Assert the model should get these right.
    assert clf.score(test_data, test_target) == 1


# %%
test_penguins(clf)


# %% [markdown]
# ## Automated Testing of Docstring Examples
#
# There is an even easier way to run simple tests. This can be useful when we write specific functions to pre-process our data.
# In the Model Sharing notebook, we looked into auto-generating docstrings.
#
# We can upgrade our docstring and get free software tests out of it!
#
# This is called doctest and usually useful to keep docstring examples up to date and write quick unit tests for a function.
#
# This makes future users (including yourself from the future) quite happy.

# %%
def shorten_class_name(df: pd.DataFrame) -> pd.DataFrame:
    """Shorten the class names of the penguins to the shortest version

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the Species column with penguins

    Returns
    -------
    pd.DataFrame
        Normalised dataframe with shortened names
    
    Examples
    --------
    >>> shorten_class_name(pd.DataFrame([[1,2,3,"Adelie Penguin (Pygoscelis adeliae)"]], columns=["1","2","3","Species"]))
       1  2  3 Species
    0  1  2  3  Adelie
    """
    df["Species"] = df.Species.str.split(r" [Pp]enguin", n=1, expand=True)[0]

    return df

import doctest
doctest.testmod()

# %%
shorten_class_name(penguins).head()

# %% [markdown]
# So these give a nice example of usage in the docstring, an expected output and a first test case that is validated by our test suite.

# %% [markdown]
# ## Input Data Validation
# You validate that the data that users are providing matches what your model is expecting.
#
# These tools are often used in production systems to determine whether APIs usage and user inputs are formatted correctly.
#
# Example tools are:
# - [Great Expectations](https://greatexpectations.io/)
# - [Pandera](https://pandera.readthedocs.io/)

# %%
import pandera as pa

# %%
# data to validate
X_train.describe()

# %%
# define schema
schema = pa.DataFrameSchema({
    "Culmen Length (mm)": pa.Column(float, checks=[pa.Check.ge(30),
                                                   pa.Check.le(60)]),
    "Culmen Depth (mm)": pa.Column(float, checks=[pa.Check.ge(13),
                                                  pa.Check.le(22)]),
    "Flipper Length (mm)": pa.Column(float, checks=[pa.Check.ge(170),
                                                    pa.Check.le(235)]),
    "Sex": pa.Column(str, checks=pa.Check.isin(["MALE","FEMALE"])),
})

validated_test = schema(X_test)


# %% [markdown]
# This fails (intentionally), because the new data is not valid.

# %%
X_test.Sex.unique()

# %%
X_test.loc[259]

# %% [markdown]
# Can you fix the data to conform to the schema?

# %%

