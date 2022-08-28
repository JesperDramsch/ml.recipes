# Contributing

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