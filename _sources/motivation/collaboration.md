# Foster Collaboration

This guide promises to foster collaboration, but initially, you may be wondering why?

We have different reasons to collaborate, to name some:

- **Access to wide-ranging Expertise and Resources:** Bringing together specialists from many domains, scientific cooperation gives researchers access to a greater range of resources and experience.
- **Higher Replicability and Better Data Quality:** Multiple researchers working on a topic provide for additional inspection of the data and methodology.
- **Increased Innovation & Creativity:**  Collaboration offers a forum for the exchange of thoughts and viewpoints and result in fresh approaches to challenging issues. 
- **Higher Productivity and Efficiency:** Quicker and more efficient achievements through pooling resources and expertise.
- **Broader Impact:** Promoting interdisciplinary research and collaboration across multiple sectors results in a broader impact on society. 

Additionally, collaborations can help to build bridges between academia and industry, leading to increased investment in scientific research and technological innovations.

But in the end the following quote holds true.

> Your closest collaborator is you from 6 months ago. And you're terrible at replying to your emails!

Here are the ways this guide fosters collaboration:

## Model Evaluation

The [guide on model evaluation](../tutorial/evaluation) ensures the validity of the machine learning model.

This sets up avenues to collaboration with domain experts to build trust between the modeller and domain scientists, who understand the caveats of their own datasets best. 
It offers a framework for evaluating and contrasting various models.
In order to verify that the findings are relevant and comparable, it is critical to evaluate models using proper metrics and criteria that have been agreed upon by ML practitioners and scientists. 

Additionally, proper model evaluation helps to identify areas for improvement and potential limitations of the models.
This can lead to further collaboration between ML practitioners and scientists to address these issues.
When model evaluation is done in a valid and transparent manner, it helps to build trust between the ML practitioners and scientists, as they are able to see the strengths and weaknesses of each other's work and collaborate to address them.

Moreover, this leads to the possible development of new evaluation metrics and techniques, which increases the quality and impact of model evaluation.
This drives innovation in the field, as new insights and techniques are generated through collaboration.

## Benchmarking
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

## Model sharing
The [guide on model sharing](../tutorial/sharing) goes over exports and best practices on code.

Model exporting allows for easy sharing and collaboration between ML practitioners and scientists. 
By exporting models in a standardized format, we can easily share their work with others, allowing for collaboration on improving and further fine-tuning of models.

Fixing sources of randomness in machine learning models ensures that the results are reproducible and comparable.
When we work together and address sources of randomness, we improve the reliability, reproducibility and robustness of models, promoting collaboration in the process.

Good code practices, such as using clear and readable code, documenting the code, and using version control, makes it easier for us to collaborate.
We can work towards accessible and usable code by others, making the training code available to collaborators and other scientists.

Fixing dependencies in our code guarantees that the models can be easily used and shared.
Working together to resolve dependencies enables us to collaborate more efficiently.

Docker is a platform that allows for easy deployment and sharing of machine learning models. 
By using Docker, we can easily share their models with others with fixed dependencies and even operating systems.

Overall, model exporting, fixing sources of randomness, good code practices, fixing dependencies, and using Docker foster collaboration by promoting transparency, reproducibility, and accessibility of machine learning models.

## Testing
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

## Interpretability
The [guide on ML interpretability](../tutorial/interpretability) provides tools for communication with domain scientists.

Scikit-learn {cite:p}`scikit-learn` provides a consistent interface to various algorithms and tools.
By using Scikit-learn, we can work together more effectively because the library provides tools for model inspection.
The library also includes tools for visualizing the performance and decision processes of models.

Tree importance and permutation importance are two methods for evaluating the feature importance in a machine learning model. 
We can have a more informed discussion with collaborators about the impact of individual features to the model's performance. 
This leads to a better understanding of the data, and helps to identify opportunities for further improvement.

SHAP (SHapley Additive exPlanations) {cite:p}`shap` is a framework for explaining the predictions of machine learning models. 
By using SHAP values, we see how each feature contributes to the model's prediction for a given sample. 
This provides insight into the workings of the model and gives a deeper understanding of the decision process.

Model inspection refers to the process of examining the internal workings of a machine learning model. 
This can help us to better understand how the model makes predictions, and can provide information about the model's strengths and weaknesses. 
By collaborating on model inspection, practitioners and scientists can work together to improve the model's performance and increase its overall accuracy.

These methods work as communication tools with other scientists.

## Ablation Studies
The [guide on ablation studies](../tutorial/ablation) works through an understanding of reducing model components.

This method builds trust in all the parts used to build a machine learning model to avoid spurious components that sneak in through the iterative nature of building data-driven solutions.

