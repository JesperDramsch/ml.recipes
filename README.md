# Increase citations, ease review & collaboration

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<YOUR URL HERE>) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JesperDramsch/ml-for-science-reproducibility-tutorial/HEAD)

A collection of "easy wins" to make machine learning in research reproducible.  

This book aims to provide easy ways to increase the quality of scientific contributions that use machine learning methods. The reproducible aspect will make it easy for fellow researchers to use and iterate on a publication, increasing citations of published work. The use of appropriate validation techniques and increase in code quality accelerates the review process during publication and avoids possible rejection due to deficiencies in the methodology. Making models, code and possibly data available increases the visibility of work and enables easier collaboration on future work.

This book focuses on basics that work. 
Getting you 90% of the way to top-tier reproducibility.

Every scientific conference has seen a massive uptick in applications that use some type of machine learning. Whether it’s a linear regression using scikit-learn, a transformer from Hugging Face, or a custom convolutional neural network in Jax, the breadth of applications is as vast as the quality of contributions.

This work to make machine learning applications reproducible has an outsized impact compared to the limited additional work that is required using existing Python libraries.

## Data

[![](https://img.shields.io/badge/view-notebook-orange)](book/notebooks/0-basic-data-prep-and-model) [![](https://img.shields.io/badge/open-colab-yellow)](https://colab.research.google.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/0-basic-data-prep-and-model.ipynb) [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/0-basic-data-prep-and-model.ipynb) [![Open%20In%20SageMaker%20Studio%20Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/jesperdramsch/ml-for-science-reproducibility-tutorial/blob/main/book/notebooks/0-basic-data-prep-and-model.ipynb)

This tutorial uses the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/).

Data were collected and made available by [Dr. Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) and the [Palmer Station, Antarctica LTER](https://pallter.marine.rutgers.edu/), a member of the [Long Term Ecological Research Network](https://lternet.edu/).

![Artwork by @allison_horst](book/img/lter_penguins.png)
*Artwork by [@allison_horst](https://www.allisonhorst.com/)*

| ▲ [Top](#increase-citations-ease-review--collaboration) |

## Usage

### Building the book

If you'd like to develop and/or build the Increase citations, ease review & collaboration book, you should:

1. Clone this repository
2. Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
3. (Optional) Edit the books source files located in the `book/` directory
4. (Optional) Jupytext syncs the content between `python_scripts` and `book/notebooks` to enable diffs.
5. Run `jupyter-book clean book/` to remove any existing builds
6. Run `jupyter-book build book/`

A fully-rendered HTML version of the book will be built in `book/_build/html/`.

### Jupytext

This repo uses: [Jupytext doc](https://jupytext.readthedocs.io/)

To synchronize the notebooks and the Python scripts (based on filestamps, only
input cells content is modified in the notebooks):

The idea and implementation for jupytext were copied from the [Euroscipy 2019 scikit-learn tutorial](https://github.com/lesteve/euroscipy-2019-scikit-learn-tutorial). Thanks for the great work!

```
$ jupytext --sync notebooks/*.ipynb
```

or simply use:

```
$ make sync
```

If you create a new notebook, you need to set-up the text files it is going to
be paired with:

```
$ jupytext --set-formats notebooks//ipynb,python_scripts//auto:percent notebooks/*.ipynb
```

or simply use:

```
$ make format
```

To render all the notebooks (from time to time, slow to run):

```
$ make render
```

| ▲ [Top](#increase-citations-ease-review--collaboration) |

### Hosting the book

Please see the [Jupyter Book documentation](https://jupyterbook.org/publish/web.html) to discover options for deploying a book online using services such as GitHub, GitLab, or Netlify.

For GitHub and GitLab deployment specifically, the [cookiecutter-jupyter-book](https://github.com/executablebooks/cookiecutter-jupyter-book) includes templates for, and information about, optional continuous integration (CI) workflow files to help easily and automatically deploy books online with GitHub or GitLab. For example, if you chose `github` for the `include_ci` cookiecutter option, your book template was created with a GitHub actions workflow file that, once pushed to GitHub, automatically renders and pushes your book to the `gh-pages` branch of your repo and hosts it on GitHub Pages when a push or pull request is made to the main branch.

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/jesperdramsch/book/graphs/contributors).

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book). Notebooks are synced with scripts using [jupytext](https://jupytext.readthedocs.io/) for version control.

| ▲ [Top](#increase-citations-ease-review--collaboration) |
