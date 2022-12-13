# Model Sharing

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/3-model-sharing) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/3-model-sharing.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/3-model-sharing.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/3-model-sharing.ipynb)

Some journals will require the sharing of code or models, but even if they don’t we might benefit from it.

Anytime we share a model, we give other researchers the opportunity to replicate our studies and iterate upon them. Altruistically, this advances science, which in and of itself is a noble pursuit. However, this also increases the citations of our original research, a core metric for most researchers in academia.

In this section, we explore how we can export models and make our training codes reproducible. Saving a model from scikit-learn is easy enough. But what tools can we use to easily make our training code adaptable for others to import and try out that model? Specifically, I want to talk about:

- Automatic Linters
- Automatic Formatting
- Automatic Docstrings and Documentation
- Docker and containerization for ultimate reproducibility

[Explore model sharing Jupyter notebook](notebooks/3-model-sharing.ipynb)
