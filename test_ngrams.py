from ngrams_model_combination import NGramsModel
from logger import Logger

if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  model = NGramsModel(n = 3, logger = logger)

  model.fit()
  model.predict()

  # model.get_best_results(topN = 5)