# Conda on Apple M1 Chip

If you're using a Mac with the latest M1 chip, it is highly recommended to install the packages in
your conda environment specifically tailored for your hardware architecture (i.e. `arm64`).
To do so, please execute the following command:

```
$ CONDA_SUBDIR=osx-arm64 conda env create -f requirements/tutorial.yml
```

This will make sure that `conda` will automatically fetch the appropriate packages from channels, if required.

To activate the environment, please run:

```
$ conda activate ml-recipes
```

Once the environment is activated, please set the `subdir` for future package installations:

```
$ conda config --env --set subdir osx-arm64
```
