# Model Evaluation

[![](https://img.shields.io/badge/view-notebook-orange)](../notebooks/1-model-evaluation) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/1-model-evaluation.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/1-model-evaluation.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/1-model-evaluation.ipynb)

Applying machine learning in an applied science context is often method work. We build a prototype model and expect want to show that this method can be applied to our specific problem. This means that we have to guarantee that the insights we glean from this application generalize to new data from the same problem set.

This is why we usually import `train_test_split()` from scikit-learn to get a validation set and a test set. But in my experience, in real-world applications, this isn’t always enough. In science, we usually deal with data that has some kind of correlation in some kind of dimension. Sometimes we have geospatial data and have to account for Tobler’s Law, i.e. things that are closer to each other matter more to each other than those data points at a larger distance. Sometimes we have temporal correlations, dealing with time series, where data points closer in time may influence each other.

Not taking care of proper validation, will often lead to additional review cycles in a paper submission. It might lead to a rejection of the manuscript which is bad enough. In the worst case scenario, our research might report incorrect conclusions and have to be retracted. No one wants rejections or even retractions.

So we’ll go into some methods to properly evaluate machine learning models even when our data is not “independent and identically distributed”.

Explore the [Jupyter notebook](../notebooks/1-model-evaluation.ipynb) on model evaluation.

Here are some of the benefits taken from the motivation section.

## Ease Scientific Review

Machine learning evaluation is an important part of the scientific review process as it helps to ensure that the results of a study are valid, reliable, and can be replicated. 
It is essential to determine the validity and generalizability of machine learning results. 
By careful evaluation and the demonstration thereof, we can disarm many criticisms during the review process.

Additionally, the ability to reproduce the results of a study is critical for scientific review and for ensuring that the results are robust and generalizable. 
This is why many researchers in the machine learning community emphasize the importance of reproducibility in their work, and why the use of clear and well-documented evaluation procedures is becoming increasingly important. 

By providing a common set of metrics and procedures for evaluating machine learning models, the scientific review process can be streamlined and made more efficient, allowing researchers to focus on the important aspects of their work, such as the design of new models and the interpretation of results.

## Foster Collaboration

The [guide on model evaluation](../tutorial/evaluation) ensures the validity of the machine learning model.

This sets up avenues to collaboration with domain experts to build trust between the modeller and domain scientists, who understand the caveats of their own datasets best. 
It offers a framework for evaluating and contrasting various models.
In order to verify that the findings are relevant and comparable, it is critical to evaluate models using proper metrics and criteria that have been agreed upon by ML practitioners and scientists. 

Additionally, proper model evaluation helps to identify areas for improvement and potential limitations of the models.
This can lead to further collaboration between ML practitioners and scientists to address these issues.
When model evaluation is done in a valid and transparent manner, it helps to build trust between the ML practitioners and scientists, as they are able to see the strengths and weaknesses of each other's work and collaborate to address them.

Moreover, this leads to the possible development of new evaluation metrics and techniques, which increases the quality and impact of model evaluation.
This drives innovation in the field, as new insights and techniques are generated through collaboration.