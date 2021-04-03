# Dynamic Topic Modeling


## Getting started

You'll need a conda environment with all the usual scientific computing packages. (Install miniconda or anaconda if you haven't already.) You can use the `environment.yml` file to generate an environment with all the packages you should need by running

```
conda env create -f environment.yml
```
The default name is `dynamic_topic_modeling`. If you would like to use a different environment name instead run
```
conda env create --name [environment_name] -f environment.yml
```

After creating the conda environment, run

```
conda activate dynamic_topic_modeling
```
You will need to reactivate the environment using the command above each time you start using the code, unless you never leave the environment.

You should now run

```
pre-commit install
```

to setup some commit hooks which will enforce a consistent coding style on the code anyone tries to commit. When committing, pre-commit will automatically run a set of style checks. It may make some automatic changes to the format of your code during these style checks, in which case you will need to re-stage the changed files and try committing again. If there are code style issues that cannot be fixed automatically, error messages will continue to appear and you will need to fix the issues manually. You can check if the commit hooks will pass without making a commit by running

```
tox -e lint
```

You should now be able to run the code. Try running

```
python
```
and then
```
>>> import covid19
```
to check that the code is setup correctly, and
```
tox
```
to make sure all of the tests pass, including the code style checks.

## Provide local path information

The scripts require knowledge of the directory that contains data and the directory in which you would like to save output figures and results. Specifically, create a file name `_local_config.py` in the scripts directory that contains:
```
data_dir = "/local/path/to/data"
results_dir = "/local/path/to/results_dir"
```

These directories are:
data_dir: the local path to the data directory.
results_dir: the local path to the desired results directory.

## Recreating experiments
Files for reproducing experimental results can be found in the `scripts` folder.

* The notebook `methods_semisynthetic_20news.ipynb` reproduces results for the semi-synthetic 20 news dataset.
* The notebook `methods_headlines_dataset.ipynb`, reproduces results for the news headlines dataset. This notebook will likely take a while (hours) to run.
* The python file `produce_twitter_figures.py` reproduces results on COVID19 related tweets. In accordance with Twitter's policies, we do not provide data for this experiment. Tweet IDs were accessed from `https://github.com/echen102/COVID-19-TweetIDs`.
* The python file `OCPDL_benchmark.py` reproduces the reconstruction error comparison plots for NCPD and Online NCPD.

## Tests

We will be using `tox` and a combination of `doctest` and `pytest` to test our code.
Doctests are written directly into the function docstrings and also provide example usage.
The pytest tests live in the `tests` directory. To execute all of the tests, run

```
tox -e py38
```
## Gotchas

* If pre-commit checks (such as the order of imports, formatting, etc.) fail, your files will likely be modified, but the modified files will remain unstaged. In order to commit successfully you will need to stage the updated files with `git add`.

* If dependencies are added to the src folder, they will need to be added to the setup.cfg file under `install_requires`. When dependencies are added to setup.cfg, run
```
tox --recreate
```
or
```
tox --recreate -e py38
```
to update tox with the new dependencies. If you add dependencies in the scripts folder or elsewhere outside of the src folder, setup.cfg does not need to be modified. Dependencies anywhere in the project (the src folder and elsewhere) should be listed in environment.yml.
