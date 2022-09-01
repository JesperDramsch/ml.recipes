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
# # Interpretability & Model Inspection
#
# One way to probe the models we build is to test them against the established knowledge of domain experts. In this final section, we’ll explore how to build intuitions about our machine learning model and avoid pitfalls like spurious correlations. These methods for model interpretability increase our trust into models, but they can also serve as an additional level of reproducibility in our research and a valuable research artefact that can be discussed in a publication.
#
# This part of the tutorial will also go into some considerations why the feature importance of tree-based methods can serve as a start but often shouldn’t be used as the sole source of truth regarding feature interpretation of our applied research.
#
# This section will introduce tools like `shap`, discuss feature importance, and manual inspection of models.

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
from joblib import dump, load

model = load("../model/svc.joblib")
model.score(X_test, y_test)

# %% [markdown]
# ## Scikit-Learn Inspection
#

# %%
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, plot_partial_dependence

pd_results = partial_dependence(model, X_train.sample(20), num_features)
pd_results


# %%
PartialDependenceDisplay.from_estimator(model, X_train, [0,1,2], target=list(y_train.unique())[0])

# %%
PartialDependenceDisplay.from_estimator(model, X_train, [0,1,2], target=list(y_train.unique())[1])

# %%
PartialDependenceDisplay.from_estimator(model, X_train, [0,1,2], target=list(y_train.unique())[2])

# %% [markdown]
# Tree importance vs Permutation importance

# %%
from sklearn.ensemble import RandomForestClassifier

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier()),
])

rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# %%
pd.Series(rf.named_steps["classifier"].feature_importances_, index=num_features+['F', 'M']).plot.bar()

# %%
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)

pd.Series(result.importances_mean, index=features).plot.bar()

# %%
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

pd.Series(result.importances_mean, index=features).plot.bar()

# %% [markdown]
# ## Shap Inspection
#

# %%
import shap

rf = RandomForestClassifier()
rf.fit(X_train[num_features], y_train)

explainer = shap.Explainer(rf)
explainer


# %%
shap_values = explainer.shap_values(X_test[num_features])

# %%
shap.initjs()

# %%
shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=num_features)

# %%
shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=num_features)

# %% [markdown]
# ## Model Inspection
#
# There are several tools that work for figuring out that a model is doing what it's supposed to do. Scikit-learn classifiers mostly work out of the box, which is why we don't necessarily have to debug the models.
#
# Sometimes we have to switch off regularization in scikit-learn to achieve the model state we expect.
#
# In neural networks we are working with many moving parts. The first step is a practical step: Overfit a small batch of data on the network. This ensures that the model is capable of learning and all the connections are made as expected. This works as a first-order sense check that models are performing.
#
# A more in-depth solution for Pytorch is [Pytorch Surgeon](https://github.com/archinetai/surgeon-pytorch), which can be used to extract submodels of the complete architecture for debugging purposes.
#
# Some example code from the Pytorch Surgeon Docs (torch and surgeon are not installed to save space):
#

# %%
import torch
import torch.nn as nn
from surgeon_pytorch import Extract, get_nodes

class SomeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 3)
        self.layer2 = nn.Linear(3, 2)
        self.layer3 = nn.Linear(1, 1)

    def forward(self, x):
        x1 = torch.relu(self.layer1(x))
        x2 = torch.sigmoid(self.layer2(x1))
        y = self.layer3(x2).tanh()
        return y

model = SomeModel()
print(get_nodes(model)) # ['x', 'layer1', 'relu', 'layer2', 'sigmoid', 'layer3', 'tanh']

# %% [markdown]
# This enables us to extract the model at one of the nodes above:

# %%
model_ext = Extract(model, node_out='sigmoid')
x = torch.rand(1, 5)
sigmoid = model_ext(x)
print(sigmoid) # tensor([[0.5570, 0.3652]], grad_fn=<SigmoidBackward0>)
