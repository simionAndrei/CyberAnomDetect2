# Cyber Fraud assignment 2 :chart_with_upwards_trend:

Code for Group 66 python implementation of Cyber Data Analytics assigment 2 CS4035. :lock:

Team members:

 * [Andrei Simion-Constantinescu](https://www.linkedin.com/in/andrei-simion-constantinescu/)
 * [Mihai Voicescu](https://github.com/mihai1voicescu)
 
## Project structure :open_file_folder:
The structure of the project is presented per task:

### Familiarization

* `Exploratory Analysis.ipynb` - jupyter notebook for explore the signals to answer the assigment's questions
* `plot_utils.py` - functions for generating bokeh and matplotlib plots
* `moving_average.py` - class implementing moving average 

### ARMA 

* `arma.py` - helper functions used for analysis
* `Arma-results.ipynb` - notebook used for the analysis
* `arma_detect_model.py` - model class and the main for running the ARMA prediction
* `arma_detect_model_residuals.py` - model class and the main for running the ARMA residuals
prediction, which presented very weak results

### Discrete model

* `discretization.py` - class implementing percentiles discretization and visualizing signal discretization
* `n_gram_model.py` - class implementing n-grams for fitting, optimizing threshold and testing on each signal
* `n_gram_model_combination.py` - Employs the same tactics as `n-gram_model.py` but additionally sorts the most relevant 5 signals according to the performance on optimization set and uses them to calculate the final prediction
* `test_ngrams.py` - main file for getting n-grams results

### PCA
* `pca_components.py` - script for determining the optimal number of components
* `pca_outliers.py` - script for plotting residuals and determining outliers from training data
* `pca_threshold.py` - script for determining threshold based on performance on optimization dataset with model fitted on training data without outliers
* `pca_test.py` - main file for getting PCA results

### BONUS
* `Bonus.ipynb` - jupyter notebook for Auto-Encoder results

#### Others
* `logger.py` -  logging system for generating folders initial structure and saving application logs to HTML files 
* `scores.py` - helper functions used to calculate the scores for the models
* `config.txt` - configuration file

#### Additional folders
* `data\` - for storing data files with BATADAL datasets
* `output\` - for storing plots at high resolution (**Better to be inspected if the ones from the report are too small due to page limit**)

## Installation :computer:
The scripts can be run in [Anaconda](https://www.anaconda.com/download/) Windows/Linux environment.

You need to create an Anaconda :snake: `python 3.6` environment named `cyber2`.
Inside that environment some addition packages needs to be installed. Run the following commands inside Anaconda Prompt ⌨:
```shell
(base) conda create -n cyber2 python=3.6 anaconda
(base) conda activate cyber2
(cyber2) conda install -c bokeh bokeh
```

For testing the **Bonus** part, another Anaconda :snake: `python 3.6` environment named cyber_keras needs to be created. Run the following commands inside Anaconda Prompt ⌨:
```shell
(base) conda create -n cyber_keras python=3.6 anaconda
(base) conda activate cyber_keras
(cyber_keras) conda install -c anaconda keras-gpu
(cyber_keras) conda install -c anaconda pydot 
```

If you wish only CPU support, replace keras installation command with *conda install -c conda-forge keras*. For GPU support, NVIDIA CUDA compatible graphic card is needed with proper drivers installed.
