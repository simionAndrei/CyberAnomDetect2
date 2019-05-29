import pandas as pd
import seaborn as sns

from logger import Logger

from arma import answer

import matplotlib.pyplot as plt

logger = Logger(show = True, html_output = True, config_file = "config.txt")
dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
train_filename = logger.config_dict['TRAIN_FILE']
logger.log("Start reading training file {}...".format(train_filename))
df_train = pd.read_csv(logger.get_data_file(train_filename), skipinitialspace = True, 
                       parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
df_train.sort_index(inplace = True)
logger.log("Finish reading training file", show_time = True)

answer(df_train, 'L_T1', 2, 0)

