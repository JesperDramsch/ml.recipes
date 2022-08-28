# Increase citations, ease review & collaboration  

A collection of "easy wins" to make machine learning in research reproducible.  

This tutorial focuses on basics that work. 
Getting you 90% of the way to top-tier reproducibility.

## Table of Contents

- [Abstract](#abstract)
- [Installation](#installation)
- [Why make it reproducible?](#why-make-it-reproducible)
- [Model Evaluation](#model-evaluation)
- [Benchmarking](#benchmarking)
- [Model Sharing](#model-sharing)
- [Testing](#testing)
- [Interpretability](#interpretability)
- [Ablation Studies](#ablation-studies)
- [Conclusion](#conclusion)

## Abstract

Every scientific conference has seen a massive uptick in applications that use some type of machine learning. Whether it’s a linear regression using scikit-learn, a transformer from Hugging Face, or a custom convolutional neural network in Jax, the breadth of applications is as vast as the quality of contributions.

This tutorial aims to provide easy ways to increase the quality of scientific contributions that use machine learning methods. The reproducible aspect will make it easy for fellow researchers to use and iterate on a publication, increasing citations of published work. The use of appropriate validation techniques and increase in code quality accelerates the review process during publication and avoids possible rejection due to deficiencies in the methodology. Making models, code and possibly data available increases the visibility of work and enables easier collaboration on future work.

This work to make machine learning applications reproducible has an outsized impact compared to the limited additional work that is required using existing Python libraries.

## Installation

Both `requirements.txt` and `environment.yml` are provided to install packages.

### Using PIP

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```
### Using Conda

You can create an `euroscipy-2022-ml-repro` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate euroscipy-2022-ml-repro
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```

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


| ▲ [Top](#table-of-contents) |

## Model Evaluation

Applying machine learning in an applied science context is often method work. We build a prototype model and expect want to show that this method can be applied to our specific problem. This means that we have to guarantee that the insights we glean from this application generalize to new data from the same problem set.

This is why we usually import `train_test_split()` from scikit-learn to get a validation set and a test set. But in my experience, in real-world applications, this isn’t always enough. In science, we usually deal with data that has some kind of correlation in some kind of dimension. Sometimes we have geospatial data and have to account for Tobler’s Law, i.e. things that are closer to each other matter more to each other than those data points at a larger distance. Sometimes we have temporal correlations, dealing with time series, where data points closer in time may influence each other.

Not taking care of proper validation, will often lead to additional review cycles in a paper submission. It might lead to a rejection of the manuscript which is bad enough. In the worst case scenario, our research might report incorrect conclusions and have to be retracted. No one wants rejections or even retractions.

So we’ll go into some methods to properly evaluate machine learning models even when our data is not “independent and identically distributed”.

[Explore model evaluation Jupyter notebook](notebooks/model-evaluation.ipynb)

| ▲ [Top](#table-of-contents) |

## Benchmarking

Another common reason for rejections of machine learning papers in applied science is the lack of proper benchmarks. This section will be fairly short, as it differs from discipline to discipline.

However, any time we apply a superfancy deep neural network, we need to supply a benchmark to compare the relative performance of our model to. These models should be established methods in the field and simpler machine learning methods like a linear model, support-vector machine or a random forest.

[Explore benchmarking Jupyter notebook](notebooks/benchmarking.ipynb)

| ▲ [Top](#table-of-contents) |

## Model Sharing

Some journals will require the sharing of code or models, but even if they don’t we might benefit from it.

Anytime we share a model, we give other researchers the opportunity to replicate our studies and iterate upon them. Altruistically, this advances science, which in and of itself is a noble pursuit. However, this also increases the citations of our original research, a core metric for most researchers in academia.

In this section, we explore how we can export models and make our training codes reproducible. Saving a model from scikit-learn is easy enough. But what tools can we use to easily make our training code adaptable for others to import and try out that model? Specifically, I want to talk about:

- Automatic Linters
- Automatic Formatting
- Automatic Docstrings and Documentation
- Docker and containerization for ultimate reproducibility

[Explore model sharing Jupyter notebook](notebooks/model-sharing.ipynb)

| ▲ [Top](#table-of-contents) |

## Testing

Machine learning is very hard to test. Due to the nature of the our models, we often have soft failures in the model that are difficult to test against.

Writing software tests in science, is already incredibly hard, so in this section we’ll touch on 

- some fairly simple tests we can implement to ensure consistency of our input data
- avoid bad bugs in data loading procedures
- some strategies to probe our models

[Explore testing Jupyter notebook](notebooks/testing.ipynb)

| ▲ [Top](#table-of-contents) |

## Interpretability

One way to probe the models we build is to test them against the established knowledge of domain experts. In this final section, we’ll explore how to build intuitions about our machine learning model and avoid pitfalls like spurious correlations. These methods for model interpretability increase our trust into models, but they can also serve as an additional level of reproducibility in our research and a valuable research artefact that can be discussed in a publication.

This part of the tutorial will also go into some considerations why the feature importance of tree-based methods can serve as a start but often shouldn’t be used as the sole source of truth regarding feature interpretation of our applied research.

This section will introduce tools like `shap`, discuss feature importance, and manual inspection of models.

[Explore interpretability Jupyter notebook](notebooks/interpretability.ipynb)

| ▲ [Top](#table-of-contents) |

## Ablation Studies

Finally, the gold standard in building complex machine learning models is proving that each constituent part of the model contributes something to the proposed solution. 

Ablation studies serve to dissect machine learning models and evaluate their impact.

In this section, we’ll finally discuss how to present complex machine learning models in publications and ensure the viability of each part we engineered to solve our particular problem set.

[Explore ablation study Jupyter notebook](notebooks/ablation-study.ipynb)

| ▲ [Top](#table-of-contents) |

## Conclusion

Overall, this tutorial is aimed at applied scientists that want to explore machine learning solutions for their problems.

This tutorial focuses on a collection of “easy wins” that scientists can implement in their research to avoid catastrophic failures and increase reproducibility with all its benefits.