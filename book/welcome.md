# Increase citations, ease review & collaboration

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JesperDramsch/ml-for-science-reproducibility-tutorial/HEAD)

A collection of "easy wins" to make machine learning in research reproducible.  

This tutorial focuses on basics that work. 
Getting you 90% of the way to top-tier reproducibility.

Every scientific conference has seen a massive uptick in applications that use some type of machine learning. Whether it’s a linear regression using scikit-learn, a transformer from Hugging Face, or a custom convolutional neural network in Jax, the breadth of applications is as vast as the quality of contributions.

This tutorial aims to provide easy ways to increase the quality of scientific contributions that use machine learning methods. The reproducible aspect will make it easy for fellow researchers to use and iterate on a publication, increasing citations of published work. The use of appropriate validation techniques and increase in code quality accelerates the review process during publication and avoids possible rejection due to deficiencies in the methodology. Making models, code and possibly data available increases the visibility of work and enables easier collaboration on future work.

This work to make machine learning applications reproducible has an outsized impact compared to the limited additional work that is required using existing Python libraries.



:::

::::

::::{grid} 1 1 2 3
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: tutorial/evaluation
:link-type: doc
:class-header: bg-light

Model Evaluation 🤖
^^^

Avoid overfitting and ensure results work on future data reliably.

:::

:::{grid-item-card}
:link: tutorial/benchmarking
:link-type: doc
:class-header: bg-light

Benchmarking 🪑
^^^

Compare your results to other solutions on standardized datasets and metrics.

:::

:::{grid-item-card}
:link: tutorial/sharing
:link-type: doc
:class-header: bg-light

Model Sharing 🤝
^^^

Export and share models to collaborate and gain citations.
:::

:::{grid-item-card}
:link: tutorial/testing
:link-type: doc
:class-header: bg-light

Testing 🧪
^^^

Catch code errors early and test that data is treated correctly.
:::

:::{grid-item-card}
:link: tutorial/interpretability
:link-type: doc
:class-header: bg-light

Interpretability ⚡
^^^

Communicate results and inspect models to avoid spurious correlations.
:::

:::{grid-item-card}
:link: tutorial/ablation
:link-type: doc
:class-header: bg-light

Ablation Studies 🔪
^^^

Model building is iterative, so explore which parts actually matter.
:::

::::


## Why make it reproducible?

One of the tenets of science is to be reproducible. 

But if we always did what we’re supposed to, the world would be a better and easier place. However, there are benefits to making science reproducible that directly benefit researchers, especially in computational science and machine learning. These benefits are like the title says:

- Easier review cycles
- More citations
- More collaboration

But it goes further. Reproducibility is a marketable skill outside of academia. When we work in companies that apply data science or machine learning, these companies know that technical debt can slowly degrade a code base and in some cases like Amazon and Google, the machine learning system has to be so reproducible that we expect the entire training and deployment to work automatically on a press of a button. Technical debt is also a problem in academia, but here it is more framed in the devastating prospect of the only postdoc leaving that knows how to operate the code base.

Luckily, we have a lot of work cut out for us already!

These benefits, and a few others, like making iteration, and therefore frequent publication easier, do not come at a proportional cost. Most of the methods to increase code quality in machine learning projects of applied scientists are in fact fairly easy to set up and run!

So how do we actually go about obtaining these goals?


| ▲ [Top](#why-make-it-reproducible) |


## Data

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/0-basic-data-prep-and-model) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/0-basic-data-prep-and-model.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/0-basic-data-prep-and-model.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/0-basic-data-prep-and-model.ipynb)

This tutorial uses the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/).

Data were collected and made available by [Dr. Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) and the [Palmer Station, Antarctica LTER](https://pallter.marine.rutgers.edu/), a member of the [Long Term Ecological Research Network](https://lternet.edu/).

![Artwork by @allison_horst](img/lter_penguins.png)
*Artwork by [@allison_horst](https://www.allisonhorst.com/)*

| ▲ [Top](#why-make-it-reproducible) |

## Model Evaluation

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/1-model-evaluation) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/1-model-evaluation.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/1-model-evaluation.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/1-model-evaluation.ipynb)

Applying machine learning in an applied science context is often method work. We build a prototype model and expect want to show that this method can be applied to our specific problem. This means that we have to guarantee that the insights we glean from this application generalize to new data from the same problem set.

This is why we usually import `train_test_split()` from scikit-learn to get a validation set and a test set. But in my experience, in real-world applications, this isn’t always enough. In science, we usually deal with data that has some kind of correlation in some kind of dimension. Sometimes we have geospatial data and have to account for Tobler’s Law, i.e. things that are closer to each other matter more to each other than those data points at a larger distance. Sometimes we have temporal correlations, dealing with time series, where data points closer in time may influence each other.

Not taking care of proper validation, will often lead to additional review cycles in a paper submission. It might lead to a rejection of the manuscript which is bad enough. In the worst case scenario, our research might report incorrect conclusions and have to be retracted. No one wants rejections or even retractions.

So we’ll go into some methods to properly evaluate machine learning models even when our data is not “independent and identically distributed”.

[Explore model evaluation Jupyter notebook](notebooks/1-model-evaluation)

| ▲ [Top](#why-make-it-reproducible) |

## Benchmarking

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/2-benchmarking) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/2-benchmarking.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/2-benchmarking.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/2-benchmarking.ipynb)

Another common reason for rejections of machine learning papers in applied science is the lack of proper benchmarks. This section will be fairly short, as it differs from discipline to discipline.

However, any time we apply a superfancy deep neural network, we need to supply a benchmark to compare the relative performance of our model to. These models should be established methods in the field and simpler machine learning methods like a linear model, support-vector machine or a random forest.

[Explore benchmarking Jupyter notebook](notebooks/2-benchmarking)

| ▲ [Top](#why-make-it-reproducible) |

## Model Sharing

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/3-model-sharing) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/3-model-sharing.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/3-model-sharing.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/3-model-sharing.ipynb)

Some journals will require the sharing of code or models, but even if they don’t we might benefit from it.

Anytime we share a model, we give other researchers the opportunity to replicate our studies and iterate upon them. Altruistically, this advances science, which in and of itself is a noble pursuit. However, this also increases the citations of our original research, a core metric for most researchers in academia.

In this section, we explore how we can export models and make our training codes reproducible. Saving a model from scikit-learn is easy enough. But what tools can we use to easily make our training code adaptable for others to import and try out that model? Specifically, I want to talk about:

- Automatic Linters
- Automatic Formatting
- Automatic Docstrings and Documentation
- Docker and containerization for ultimate reproducibility

[Explore model sharing Jupyter notebook](notebooks/3-model-sharing)

| ▲ [Top](#why-make-it-reproducible) |

## Testing

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/4-testing) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/4-testing.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/4-testing.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/4-testing.ipynb)

Machine learning is very hard to test. Due to the nature of the our models, we often have soft failures in the model that are difficult to test against.

Writing software tests in science, is already incredibly hard, so in this section we’ll touch on 

- some fairly simple tests we can implement to ensure consistency of our input data
- avoid bad bugs in data loading procedures
- some strategies to probe our models

[Explore testing Jupyter notebook](notebooks/4-testing)

| ▲ [Top](#why-make-it-reproducible) |

## Interpretability

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/5-interpretability) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/5-interpretability.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/5-interpretability.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/5-interpretability.ipynb)

One way to probe the models we build is to test them against the established knowledge of domain experts. In this final section, we’ll explore how to build intuitions about our machine learning model and avoid pitfalls like spurious correlations. These methods for model interpretability increase our trust into models, but they can also serve as an additional level of reproducibility in our research and a valuable research artefact that can be discussed in a publication.

This part of the tutorial will also go into some considerations why the feature importance of tree-based methods can serve as a start but often shouldn’t be used as the sole source of truth regarding feature interpretation of our applied research.

This section will introduce tools like `shap`, discuss feature importance, and manual inspection of models.

[Explore interpretability Jupyter notebook](notebooks/5-interpretability)

| ▲ [Top](#why-make-it-reproducible) |

## Ablation Studies

[![](https://img.shields.io/badge/view-notebook-orange)](notebooks/6-ablation-study) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/6-ablation-study.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/6-ablation-study.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/6-ablation-study.ipynb)

Finally, the gold standard in building complex machine learning models is proving that each constituent part of the model contributes something to the proposed solution. 

Ablation studies serve to dissect machine learning models and evaluate their impact.

In this section, we’ll finally discuss how to present complex machine learning models in publications and ensure the viability of each part we engineered to solve our particular problem set.

[Explore ablation study Jupyter notebook](notebooks/6-ablation-study)

| ▲ [Top](#why-make-it-reproducible) |
## Conclusion

Overall, this tutorial is aimed at applied scientists that want to explore machine learning solutions for their problems.

This tutorial focuses on a collection of “easy wins” that scientists can implement in their research to avoid catastrophic failures and increase reproducibility with all its benefits.
