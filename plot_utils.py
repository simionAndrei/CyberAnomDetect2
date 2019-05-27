from bokeh.plotting import figure, output_file, save

from bokeh.models import CustomJS, ColumnDataSource, Select, CheckboxGroup
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def create_water_tanks_iplot(data_df, logger):

  columns = ['L_T' + str(i) for i in range(1, 8)]
  max_y = max([data_df[col].max() for col in columns])

  source = ColumnDataSource({
    'x_t1' : data_df.index, 'y_t1' : data_df['L_T1'],
    'x_t2' : data_df.index, 'y_t2' : data_df['L_T2'],
    'x_t3' : data_df.index, 'y_t3' : data_df['L_T3'],
    'x_t4' : data_df.index, 'y_t4' : data_df['L_T4'],
    'x_t5' : data_df.index, 'y_t5' : data_df['L_T5'],
    'x_t6' : data_df.index, 'y_t6' : data_df['L_T6'],
    'x_t7' : data_df.index, 'y_t7' : data_df['L_T7'],
  })

  p = figure(plot_width=1000, plot_height=750,
      title="Water tanks", tools = "pan,wheel_zoom,save,reset",
      x_axis_type="datetime", y_range=[0, max_y + 2.5],
      x_axis_label = "Time",
      y_axis_label = "Water level (m)")
  p.xaxis.axis_label_text_font_size = "18pt";
  p.yaxis.axis_label_text_font_size = "18pt";

  p_t1 = p.line('x_t1', 'y_t1', source=source, color='red', 
    alpha=0.9, line_width = 2, legend = "Tank1")
  p_t2 = p.line('x_t2', 'y_t2', source=source, color='gold',
    alpha=0.9, line_width = 2, legend = "Tank2")
  p_t3 = p.line('x_t3', 'y_t3', source=source, color='blue',
    alpha=0.9, line_width = 2, legend = "Tank3")
  p_t4 = p.line('x_t4', 'y_t4', source=source, color='black',
    alpha=0.9, line_width = 2, legend = "Tank4")
  p_t5 = p.line('x_t5', 'y_t5', source=source, color='mediumspringgreen',
    alpha=0.9, line_width = 2, legend = "Tank5")
  p_t6 = p.line('x_t6', 'y_t6', source=source, color='salmon',
    alpha=0.9, line_width = 2, legend = "Tank6")
  p_t7 = p.line('x_t7', 'y_t7', source=source, color='navy',
    alpha=0.9, line_width = 2, legend = "Tank7")

  p.legend.location = "top_left"

  tanks = ['Tank ' + str(i) for i in range(1, 8)]
  checkboxes = CheckboxGroup(labels=tanks, active=[i for i in range(7)])
  callback = CustomJS(args=dict(p_t1=p_t1, p_t2=p_t2, p_t3=p_t3, p_t4=p_t4, 
    p_t5=p_t5, p_t6=p_t6, p_t7=p_t7), 
    code="""
    f = cb_obj.active;
    p_t1.visible = false;
    p_t2.visible = false;
    p_t3.visible = false;
    p_t4.visible = false;
    p_t5.visible = false;
    p_t6.visible = false;
    p_t7.visible = false;
    if (f.includes(0)) { p_t1.visible = true; }
    if (f.includes(1)) { p_t2.visible = true; }
    if (f.includes(2)) { p_t3.visible = true; }
    if (f.includes(3)) { p_t4.visible = true; }
    if (f.includes(4)) { p_t5.visible = true; }
    if (f.includes(5)) { p_t6.visible = true; }
    if (f.includes(6)) { p_t7.visible = true; }
    """
  )
  checkboxes.js_on_click(callback)

  output_file(logger.get_output_file("tanks.html"))
  save(row(checkboxes, p))


def create_correlation_heatmap(plt_title, corr_df, feats_names, filename, logger):

  fig = plt.figure(figsize=(9, 9))

  colormap = sns.diverging_palette(220, 10, as_cmap=True)
  ax = sns.heatmap(corr_df, cmap = colormap) #, annot = True, fmt = ".2f")
  ax.set_title(plt_title)

  x = np.array(range(len(feats_names)))
  plt.xticks(x, feats_names)
  plt.yticks(x, feats_names)

  plt.savefig(logger.get_output_file(filename), dpi = 120, 
    bbox_inches='tight')


def create_pump_station_plot(data_df, filename, logger):

  sns.set()
  plt.figure(figsize=(12,7))
  ax = plt.subplot(111)

  plt.title("Pumping station 1")
  plt.xlabel("Time")
  plt.ylabel("Flow / Status")
  plt.plot(data_df.index, data_df['F_PU1'], color = 'red')
  plt.plot(data_df.index, data_df['F_PU2'], color = 'gold')
  plt.plot(data_df.index, data_df['F_PU3'], color = 'blue')
  plt.plot(data_df.index, data_df['S_PU1'] * 110, color = 'black')
  plt.plot(data_df.index, data_df['S_PU2'] * 30, color = 'salmon')
  plt.plot(data_df.index, data_df['S_PU3'] * 20, color = 'navy')
 
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  legend_str  =  ["Flow for Pump " + str(i) for i in range(1, 4)]
  legend_str +=  ["Status for Pump " + str(i) for i in range(1, 4)]
  plt.legend(legend_str, loc = 'center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)
  plt.savefig(logger.get_output_file(filename), dpi = 120, 
    bbox_inches='tight')


def create_water_tanks_plot(data_df, columns, filename, logger):

  sns.set()
  plt.figure(figsize=(12,6))
  ax = plt.subplot(111)

  plt.title("Water tanks level")
  plt.xlabel("Time")
  plt.ylabel("Level (m)")

  colors = ["red", "gold", "blue", "black", "salmon", "navy", "mediumspringgreen"]
  for column, color in zip(columns, colors):
    plt.plot(data_df.index, data_df[column], color = color)

  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  legend_str  = [e.split("_")[-1][-1] for e in columns]
  legend_str  =  ["Tank " + str(i) for i in legend_str]
  ncols = 2 if len(columns) > 3 else 1
  plt.legend(legend_str, loc = 'center left', bbox_to_anchor=(1, 0.5), 
    ncol = ncols, fancybox=True, shadow=True)
    
  plt.savefig(logger.get_output_file(filename), dpi = 120, 
    bbox_inches='tight')


def visualize_attack(times, normal_values, attack_values, title, axis_label, 
  filename, logger):

  sns.set()
  plt.figure(figsize=(10,5))
  ax = plt.subplot(111)

  plt.plot(times, normal_values, color = 'blue')
  plt.plot(times, attack_values, color = 'red')

  plt.title(title)
  plt.xlabel("Time")
  plt.ylabel(axis_label)

  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  legend_str  =  ["Normal signal", "Signal during attack"]
  plt.legend(legend_str, loc = 'center left', bbox_to_anchor=(1, 0.5), 
    fancybox=True, shadow=True)
  plt.savefig(logger.get_output_file(filename), dpi = 120, 
    bbox_inches='tight')







  






  


'''
from logger import Logger

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = Logger(show = True, html_output = True, config_file = "config.txt")

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
train_filename = logger.config_dict['TRAIN_FILE']
df_train = pd.read_csv(logger.get_data_file(logger.config_dict['TRAIN_FILE']), 
                       parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
df_train.sort_index(inplace = True)

df_train_days = df_train.groupby(pd.Grouper(freq='D')).mean()
from plot_utils import make_water_tanks_plot
make_water_tanks_plot(df_train_days, logger)
'''