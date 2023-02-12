# Benchmarking

[![](https://img.shields.io/badge/view-notebook-orange)](../notebooks/2-benchmarking) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/2-benchmarking.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/2-benchmarking.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/2-benchmarking.ipynb)

Another common reason for rejections of machine learning papers in applied science is the lack of proper benchmarks. This section will be fairly short, as it differs from discipline to discipline.

However, any time we apply a superfancy deep neural network, we need to supply a benchmark to compare the relative performance of our model to. These models should be established methods in the field and simpler machine learning methods like a linear model, support-vector machine or a random forest.

Explore the [Jupyter notebook](../notebooks/2-benchmarking.ipynb) on benchmarking.

Here are some of the benefits taken from the motivation section.

## Increase Citations

Machine learning benchmarking increases citations of scientific work by providing a standardized way to evaluate and compare different machine learning models on a specific task.
This allows researchers to compare their models to others in the field, and demonstrate their model's performance in a standardized, transparent and reproducible way.

By participating in machine learning benchmarking, researchers can showcase the strengths and weaknesses of their models, and provide evidence of their model's performance relative to other scientists. 
This helps to increase visibility and credibility of their work, and leads to increased citations. 
For instance, a machine learning model that performs well on a benchmarked task is more likely to be noticed by others in the field, and may be more likely to be used by other researchers as a baseline for their own work.

Additionally, participating in benchmarking can also provide opportunities for collaboration, as researchers may identify areas for improvement in their models and work together to address these issues. 
This leads to increased citations as the improved models are recognized and cited by others in the field. 
Overall, machine learning benchmarking helps increase citations of scientific work by providing a standardized way to evaluate and compare models, increasing visibility and credibility, and providing opportunities for collaboration.

## Foster Collaboration

The [guide on benchmarking](../tutorial/benchmarking) outlines different ways to anchor results from ML models.

Dummy models are simple and straightforward models that serve as a baseline for comparison with more complex models. 
By comparing the performance of a complex model with a dummy model, ML practitioners and scientists can better understand the added value of the complex model and identify areas for improvement. 
This fosters collaboration through grounding our models in a lower bound as the random statistical equivalence.

Benchmark datasets provide a standard set of data that can be used to evaluate the performance of machine learning models.
That means we can compare their models with those of other ML practitioners and domain scientists in the field.

Domain methods refer to specific techniques or methods that are commonly used in a particular field or application area.
When using domain methods, we can better understand the specific requirements and challenges of a particular application area and collaborate to develop new and innovative solutions.

Linear and simple models serve as a starting point for more complex models.
We can gain a deeper understanding of the data and problem by applying linear models.
This fosters collaboration by grounding our work in the simplest model and enables comparison with a baseline.

These models can all play a role in fostering collaboration between ML practitioners and scientists by providing common standards, baselines, and starting points for model evaluation and improvement.
