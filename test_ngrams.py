from ngrams_model import NGramsModel
from logger import Logger

if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  model = NGramsModel(n = 2, logger = logger)

  model.fit()
  model.predict()