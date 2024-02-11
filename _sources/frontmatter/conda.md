# Using Conda

⚠️ If you're using Apple with M1 Chip, please follow these [instructions](/frontmatter/conda-m1.html)

You can create an `ml-recipes` conda environment executing:

```
$ conda env create -f requirements/tutorial.yml
```

and later activate the environment:

```
$ conda activate ml-recipes
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file requirements/tutorial.yml  --prune
```
