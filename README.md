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
* `moving_average.py` - 

* `arma.py` - helper functions used for analysis
* `Arma-results.ipynb` - notebook used for the analysis
* `arma_detect_model.py` - model class and the main for running the ARMA prediction
* `arma_detect_model_residuals.py` - model class and the main for running the ARMA residuals
prediction, which presented very weak results
* `scores.py` - helper functions used to calculate the scores for the models
* `n_gram_model_combination.py` - Employs the same tactics as `n-gram_model.py` but additionally sorts the most relevant 5 signals from the optimization set and uses it to calculate the final prediction
