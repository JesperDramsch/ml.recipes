# Testing

[![](https://img.shields.io/badge/view-notebook-orange)](../notebooks/4-testing) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/4-testing.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/4-testing.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/4-testing.ipynb)

Machine learning is very hard to test. Due to the nature of the our models, we often have soft failures in the model that are difficult to test against.

Writing software tests in science, is already incredibly hard, so in this section we’ll touch on 

- some fairly simple tests we can implement to ensure consistency of our input data
- avoid bad bugs in data loading procedures
- some strategies to probe our models

Explore the [Jupyter notebook](../notebooks/4-testing.ipynb) on testing.

Here are some of the benefits taken from the motivation section.

## Ease Scientific Review

Machine learning testing plays a crucial role in easing the scientific review process by providing a means to evaluate the validity and reliability of the results reported in a research paper. 
Testing can help to determine whether a machine learning model has been implemented and trained correctly, and whether it produces accurate and consistent results.

Testing can also provide a way to verify the claims made in a research paper, such as the model's accuracy, performance, and scalability. 
This means we can identify potential errors, limitations, and weaknesses in the model and the experimental design, which can be addressed during the review process.

In summary, machine learning testing helps the scientific review process to be more rigorous, transparent, and objective, and that the results reported in a research paper are accurate and reliable. 
This, in turn, increases the impact and influence of the research, and ultimately contributes to advancing science with machine learning.

## Increase Citations

Machine learning testing increases citations of scientific work by providing a rigorous evaluation of the performance and reliability of machine learning models. 
The use of well-designed testing procedures can ensure the validity and accuracy of a model, which in turn can increase its visibility and credibility among the scientific community. 

When a machine learning model is thoroughly tested, it is easier to use pre-existing solutions and reproduce. 
Other researchers may use it as a basis for comparison and further development. 

Furthermore, testing can also identify areas where a model may have limitations or weaknesses, allowing researchers to address these issues and improve the model, which provides a clear and objective evaluation of machine learning models and research code.

## Foster Collaboration

The [guide on testing machine learning](../tutorial/testing) works through easy ways to ensure consistent processing of data and methods.

Deterministic tests enable us to know that the results of machine learning models are consistent and predictable, even when the underlying data changes.
We conduct deterministic tests to identify and fix any issues in the models, promoting the reliability and robustness of the models.
This builds trust with other practitioners and safeguards that changes to the code don't introduce bugs in custom methods.

Data tests for models test that the models produce the correct output on known standard examples. 
These tests are essential when working with domain scientists who know that certain data points work as a canary.

Similarly, automated testing of docstrings helps to ensure that the documentation of the models is accurate and up-to-date. 
This automated testing of the documentation is one of the simplest form of implementing tests that promotes the transparency and accessibility of machine learning models and methods.

Input data validation takes the other path, so that the models are only applied to appropriate data, preventing any potential issues or errors.
This input data validation implements reliability and robustness of the models, when we hand off a model to collaborators.

Deterministic and data tests, automated tests of docstrings, and input data validation foster collaboration between machine learning practitioners and domain scientists by promoting reliability, reproducibility, robustness, transparency, and trust in the models, allowing for easy and effective collaboration on their development and improvement.
