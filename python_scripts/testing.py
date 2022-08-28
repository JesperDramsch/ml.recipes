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
