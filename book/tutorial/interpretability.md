# Interpretability

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/5-interpretability) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/5-interpretability.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/5-interpretability.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/5-interpretability.ipynb)

One way to probe the models we build is to test them against the established knowledge of domain experts. In this final section, we’ll explore how to build intuitions about our machine learning model and avoid pitfalls like spurious correlations. These methods for model interpretability increase our trust into models, but they can also serve as an additional level of reproducibility in our research and a valuable research artefact that can be discussed in a publication.

This part of the tutorial will also go into some considerations why the feature importance of tree-based methods can serve as a start but often shouldn’t be used as the sole source of truth regarding feature interpretation of our applied research.

This section will introduce tools like `shap`, discuss feature importance, and manual inspection of models.

[Explore interpretability Jupyter notebook](notebooks/5-interpretability.ipynb)