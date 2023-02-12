# Interpretability

[![](https://img.shields.io/badge/view-notebook-orange)](../notebooks/5-interpretability) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/5-interpretability.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/5-interpretability.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/5-interpretability.ipynb)

One way to probe the models we build is to test them against the established knowledge of domain experts. In this final section, we’ll explore how to build intuitions about our machine learning model and avoid pitfalls like spurious correlations. These methods for model interpretability increase our trust into models, but they can also serve as an additional level of reproducibility in our research and a valuable research artefact that can be discussed in a publication.

This part of the tutorial will also go into some considerations why the feature importance of tree-based methods can serve as a start but often shouldn’t be used as the sole source of truth regarding feature interpretation of our applied research.

This section will introduce tools like `shap`, discuss feature importance, and manual inspection of models.

Explore the [Jupyter notebook](../notebooks/5-interpretability.ipynb) on interpretability.

Here are some of the benefits taken from the motivation section.

## Ease Scientific Review

Machine learning interpretability refers to the ease with which the workings and decisions of a machine learning model can be understood by human experts. 
In the scientific review process, interpretability of machine learning models is important because it allows reviewers to evaluate the validity and reliability of the model, and to assess its limitations and potential biases.

When a machine learning model is interpretable, reviewers can understand how the model is making predictions, what factors it is taking into account, and how it is combining these factors to reach its conclusions. 
This makes it easier for reviewers to assess the quality of the model, and to identify areas where further improvement or validation is needed.

Furthermore, interpretability can also help reviewers to understand the assumptions and limitations of the model, and to detect any potential biases or errors in its design or implementation. 
This can prevent reviewers from accepting models that are unreliable, flawed, or biased, and can help to ensure that only models of high quality are accepted for publication.

Overall, explainable AI is a crucial factor in the scientific review process for machine learning models, as it helps to increase the transparency, reliability, and validity of these models, and to ensure that the results they produce are trustworthy and reproducible.

## Foster Collaboration

The [guide on ML interpretability](../tutorial/interpretability) provides tools for communication with domain scientists.

Scikit-learn provides a consistent interface to various algorithms and tools. 
By using Scikit-learn, we can work together more effectively because the library provides tools for model inspection.
The library also includes tools for visualizing the performance and decision processes of models.

Tree importance and permutation importance are two methods for evaluating the feature importance in a machine learning model. 
We can have a more informed discussion with collaborators about the impact of individual features to the model's performance. 
This leads to a better understanding of the data, and helps to identify opportunities for further improvement.

SHAP (SHapley Additive exPlanations) is a framework for explaining the predictions of machine learning models. 
By using SHAP values, we see how each feature contributes to the model's prediction for a given sample. 
This provides insight into the workings of the model and gives a deeper understanding of the decision process.

Model inspection refers to the process of examining the internal workings of a machine learning model. 
This can help us to better understand how the model makes predictions, and can provide information about the model's strengths and weaknesses. 
By collaborating on model inspection, practitioners and scientists can work together to improve the model's performance and increase its overall accuracy.

These methods work as communication tools with other scientists.

