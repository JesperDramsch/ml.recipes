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
# # Ablation Studies
#
# Finally, the gold standard in building complex machine learning models is proving that each constituent part of the model contributes something to the proposed solution. 
#
# Ablation studies serve to dissect machine learning models and evaluate their impact.
#
# In this section, we’ll finally discuss how to present complex machine learning models in publications and ensure the viability of each part we engineered to solve our particular problem set.

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
import numpy as np
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score

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
model

# %%
scores = cross_val_score(model, X_test, y_test, cv=10)


# %% [markdown]
# Let's compare this to a model that doesn't scale the numeric inputs. 

# %%
scoring = pd.DataFrame(columns=["Average", "Deviation"])
scoring.loc["Full", :] = [scores.mean(), scores.std()]
scoring

# %%

# num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    # ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])


model2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC()),
])

scores = cross_val_score(model2, X_test, y_test, cv=10)

scoring.loc["No Standardisation",:] = [scores.mean(), scores.std()]

# %%
scoring

# %%

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])


model2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC()),
])

scores = cross_val_score(model2, X_test, y_test, cv=10)

scoring.loc["Single Column Sex",:] = [scores.mean(), scores.std()]
scoring

# %% [markdown]
# We can see that the standardization has an important effect on our model performance.
#
# Ideally, we would switch components of the final model of iteratively to obtain the individual impact of each component.
# This works best, if the score initially doesn't finish at 100%, but we can still see if anything is catastrophic with regards to the model performance.
#
# Simply using cross-validation is the simplest way. Alternatively, model-selection metrics like the AIC or BIC can be appropriate to evaluate the actual information that is processed by the model.