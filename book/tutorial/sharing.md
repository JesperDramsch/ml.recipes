# Model Sharing

[![](https://img.shields.io/badge/view-notebook-orange)](../notebooks/3-model-sharing) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/3-model-sharing.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/3-model-sharing.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/notebooks/3-model-sharing.ipynb)

Some journals will require the sharing of code or models, but even if they don’t we might benefit from it.

Anytime we share a model, we give other researchers the opportunity to replicate our studies and iterate upon them. Altruistically, this advances science, which in and of itself is a noble pursuit. However, this also increases the citations of our original research, a core metric for most researchers in academia.

In this section, we explore how we can export models and make our training codes reproducible. Saving a model from scikit-learn is easy enough. But what tools can we use to easily make our training code adaptable for others to import and try out that model? Specifically, I want to talk about:

- Automatic Linters
- Automatic Formatting
- Automatic Docstrings and Documentation
- Docker and containerization for ultimate reproducibility

Explore the [Jupyter notebook](../notebooks/3-model-sharing.ipynb) on model sharing.

Here are some of the benefits taken from the motivation section.

## Increase Citations

Sharing machine learning models can help increase the citations of scientific work in several ways:

- **Reproducibility:** Sharing machine learning models increases the transparency and reproducibility of scientific work. This is because other scientists can easily use shared models to confirm the results and replicate experiments, which increases confidence in the validity of the findings. This sets up to increased recognition of the original authors' work.

- **Improved Model Quality:** Sharing machine learning models also provides opportunities for other scientists to improve the models and extend the work. Researchers then suggest modifications or extensions to a model. These contributions increase the visibility and impact of the original authors' work, and result in additional citations.

- **Broader Impact:** By sharing machine learning models, researchers can make their work more accessible to a wider audience. This includes researchers from different fields, practitioners in industry, and even the general public. The wider dissemination of the work increases awareness and understanding of the research, leading to citations.

Overall, sharing machine learning models can help increase the citations of scientific work by promoting transparency, reproducibility, improved model quality, and broader impact.

## Foster Collaboration

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
