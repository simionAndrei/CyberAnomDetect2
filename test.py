import pandas as pd
import seaborn as sns

from logger import Logger

from arma import answer

import matplotlib.pyplot as plt

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
logger = Logger(show = True, html_output = True, config_file = "config.txt")
test_filename = logger.config_dict['TEST_FILE']
logger.log("Start reading testing file {}...".format(test_filename))
df_test = pd.read_csv(logger.get_data_file(test_filename), skipinitialspace = True,
    parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
df_test.sort_index(inplace = True)
logger.log("Finish reading test file", show_time = True)

df_test['ATT_FLAG'] = [0 for _ in range(df_test.shape[0])]

df_test['ATT_FLAG'].loc['2017-01-16 09:00:00':'2017-01-19 06:00:00'] = 1
df_test['ATT_FLAG'].loc['2017-01-30 08:00:00':'2017-02-02 00:00:00'] = 1
df_test['ATT_FLAG'].loc['2017-02-09 03:00:00':'2017-02-10 09:00:00'] = 1
df_test['ATT_FLAG'].loc['2017-02-12 01:00:00':'2017-02-13 07:00:00'] = 1
df_test['ATT_FLAG'].loc['2017-02-24 05:00:00':'2017-02-28 08:00:00'] = 1
df_test['ATT_FLAG'].loc['2017-03-10 14:00:00':'2017-03-13 21:00:00'] = 1
df_test['ATT_FLAG'].loc['2017-03-25 20:00:00':'2017-03-27 01:00:00'] = 1

df_test['DATETIME'] = df_test.index.values

columns = ['DATETIME']
columns += df_test.columns[:-1].tolist()


df_test = df_test[columns] 

print("Here")
df_test.to_csv(logger.get_data_file("BATADAL_test_dataset_labeled.csv"), index = False)