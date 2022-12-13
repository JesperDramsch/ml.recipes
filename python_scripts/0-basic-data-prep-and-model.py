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
# # Getting to know the data
#
# This tutorial uses the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/).
#
# Data were collected and made available by [Dr. Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) and the [Palmer Station, Antarctica LTER](https://pallter.marine.rutgers.edu/), a member of the [Long Term Ecological Research Network](https://lternet.edu/).
#
# Let's dive into some quick exploration of the data!

# %%
from pathlib import Path

DATA_FOLDER = Path("..") / "data"
DATA_FILEPATH = DATA_FOLDER / "penguins.csv"

# %%
import pandas as pd

# %%
penguins_raw = pd.read_csv(DATA_FILEPATH)
penguins_raw.head()

# %% [markdown]
# This looks like a lot. Let's reduce this to some numerical columns and the species as our target column.

# %%
num_features = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]
cat_features = ["Sex"]
features = num_features + cat_features
target = ["Species"]
penguins = penguins_raw[features+target]
penguins

# %% [markdown]
# ## Data Visualization
#
# That's much better. So now we can look at the data in detail.

# %%
import seaborn as sns

pairplot_figure = sns.pairplot(penguins, hue="Species")

# %% [markdown]
# ## Data cleaning
#
# Looks like we're getting some good separation of the clusters. 
#
# So that means we can probably do some cleaning and get ready to build some good machine learning models.

# %%
penguins = penguins.dropna(axis='rows')
penguins

# %%
DATA_CLEAN_FILEPATH = DATA_FOLDER / "penguins_clean.csv"

penguins.to_csv(DATA_CLEAN_FILEPATH, index=False)

# %% [markdown]
# Not too bad it looks like we lost two rows. That's manageable, it's a toy dataset after all.
#
# So let's build a small model to classify penuins!
# ## Machine Learning
#
# First we need to split the data.
#
# This way we can test whether our model learned general rules about our data, or if it just memorized the training data.
# When a model does not learn the generalities, this is known as overfitting.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(penguins[features], penguins[target], train_size=.7)
X_train

# %%
y_train

# %% [markdown]
# Now we can build a machine learning model.
#
# Here we'll use the scikit-learn pipeline model. 
# This makes it really easy for us to train prepocessors and models on the training data alone and cleanly apply to the test data set without leakage.
#
# ### Pre-processing
# <div class="alert alert-block alert-info">
# <b>Tip:</b> In science, any type of feature selection, scaling, basically anything you do to the data, needs to be done <strong>after</strong> a split into training and test set.<br>Statistically and scientifically valid results come from proper treatment of our data. Unfortunately, we can overfit manually if we don't split out a test set before pre-processing.</div>

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# %% [markdown]
# The `ColumnTransformer` is a neat tool that can apply your preprocessing steps to the right columns in your dataset. 
#
# In fact, you could also use a Pipeline instead of "just" a `StandardScaler` to use more sophisticated and complex preprocessing workflows that go beyond this toy project.

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# %%
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC()),
])
model

# %% [markdown]
# We can see a nice model representation here.
#
# You can click on the different modules that will tell you which arguments were passed into the pipeline. In our case, how we handle unknown values in the OneHotEncoder.
#
# ### Model Training
# Now it's time to fit our Support-Vector Machine to our training data.

# %%
model.fit(X_train, y_train[target[0]])

# %% [markdown]
# We can see that we get a decent score on the training data.
#
# This metric only tells us how well the model can perform on the data it has seen, we don't know anything about generalization and actual "learning" yet.

# %%
model.score(X_train, y_train)

# %% [markdown]
# To evaluate how well our model learned, we check the model against the test data one final time.
# <div class="alert alert-block alert-info">
# <b>Tip:</b> It is possible to manually overfit a model to the test set, by tweaking the pipelines based on the test score.<br>This invalidates scientific results and must be avoided. Only evaluate on the test set once!</div>


# %%
model.score(X_test, y_test)
