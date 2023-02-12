# Using Conda

⚠️ If you're using Apple with M1 Chip, please follow these [instructions](#note-for-conda-on-apple-m1-chip)

You can create an `pydata-global-2022-ml-repro` conda environment executing:

```
$ conda env create -f requirements/tutorial.yml
```

and later activate the environment:

```
$ conda activate pydata-global-2022-ml-repro
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```
