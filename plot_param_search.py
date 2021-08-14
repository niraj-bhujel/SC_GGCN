import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from IPython import get_ipython

def plot_line(data, parameters,  metric='test_fde'):
    '''
    Line plot of parameter values and correspnding metric values. It allows to see 
    minimum and maximum point at which the error is minimum. 
    
    ----------
    data : data frame containing headers and run results
    parameters : list of parameters to plot e.g. ['batch_size', 'hidden_size']
    metrics : metric name to plot
    '''
    fig, axes = plt.subplots(nrows=len(parameters), ncols=1)
    for i, param in enumerate(parameters):
        #sort data by  parameter values and metric as parameter values may be same across multiple rows 
        sorted_data = data.sort_values([param, metric]) 
        ax = axes[i]
        ax.plot(sorted_data[param], sorted_data[metric],'r-', label=metric)
        ax.set_title(param + ' vs'+ metric)
        ax.set_xticks(list(np.unique(data[param])))
        # ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid()
    
def plot_error_bar(data, parameters, metric='test_fde'):
    '''
    Parameters
    ----------
    data : Panda  data frame with headers
        DESCRIPTION.
    parameters : list of parameters to plot e.g. ['batch_size', 'hidden_size']
        DESCRIPTION.

    '''
    tmp_dic = {0:'a', 1:'b', 2:'c', 3:'d'}
    fig, axes = plt.subplots(nrows=len(parameters), ncols=1, sharey=True, figsize=(5, 8))
    for i, param in enumerate(parameters):
        uniq_values = np.unique(data[param])
        
        #compute means, stds, maxes and mins for each parameter        
        means = np.zeros(len(uniq_values))
        stds  = np.zeros(len(uniq_values))
        mins = np.zeros(len(uniq_values))
        maxes = np.zeros(len(uniq_values))
        
        for j, val in enumerate(uniq_values):
            metric_val = data.loc[data[param] == val, metric]
            means[j] = metric_val.mean(0)
            stds[j] = metric_val.std(0)
            mins[j] = metric_val.min(0)
            maxes[j] = metric_val.max(0)
        
        #plot error bar for each parameter
        ax = axes[i]
        ax.errorbar(uniq_values, means, stds, fmt='ok', lw=3)
        ax.errorbar(uniq_values, means, [means - mins, maxes - means],
                     fmt='.k', ecolor='gray', lw=1)
        # ax.set_title(param +' vs ' + metric)
        ax.set_title('(%s) %s'%(tmp_dic[i], param))
        ax.set_xticks(list(uniq_values))
        # ax.set_xlabel(param)
        # ax.set_ylabel('error')
        # ax.set_xscale('log')
        ax.grid()
        
        #plot min errors 
        # ax.plot(uniq_values, mins,'c--', marker='*', label='min_{}'.format(metric))
        line, = ax.plot(uniq_values, means,'m--', marker='*', label='Avg_{}'.format(metric))
        # ax.legend()
    fig.text(0.04, 0.5, 'Validation ADE', va='center', rotation='vertical')
    fig.legend([line], ['Test ADE'])
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
def plot_line_bar(data, parameters, metric='test_fde'):
    '''
    Plot error in line and bar format for each run or model

    data : panda data frame with header and values
    parameters : parameters to plot e.g ['batch_size', 'hidden_size']
    metric : which metric to plot
    '''
    #plt.close('all')       
    x_label_list=[]
    for idx, row in data.iterrows():
        x_labels=', '.join(['{}'.format(row[p]) for p in parameters])
        x_label_list.append(x_labels)

    fig, ax = plt.subplots()
    
    run = np.arange(len(data))
    values = data.loc[:, metric]
    
    #plot bar
    ax.bar(run, values, 0.7, color="grey")
    
    #plot line
    ax.plot(run, values, 'r-')
    
    #annotate
    for i,j in zip(run, values):
        ax.annotate('{:.2f}'.format(j), xy=(i, j+0.05))
    
    ax.set_xticks(run)
    ax.set_xticklabels(x_label_list, rotation=45, fontsize=8, ha='right')
    minorLocator = MultipleLocator(1)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.grid(which='minor')
    ax.set_xlabel(', '.join(parameters))
    ax.set_ylabel(metric)
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

file_path='./out/SC_GCN/log_results_trial_13.xlsx'
if '.csv' in file_path:
    #read file header
    with open(file_path, 'r') as f:
        all_data = [line.strip() for line in f.readlines()]
        header = all_data[0].split(',')
        data = []
        for line in all_data[1:]:
            data.append([float(x) for x in line.split(',')])
        data = np.array(data)
        
if '.xlsx' in file_path:
    panda_df = pd.read_excel(file_path, sheet_name='Sheet')

# panda_df = panda_df.loc[panda_df['hidden_dim'] == 64]
# panda_df = panda_df.loc[panda_df['batch_size'] == 64]
# panda_df = panda_df.loc[panda_df['dataset'] == 'univ']

parameters = ['z_dim', 'enc_layers', 'dec_layers']

plt.close('all')
# get_ipython().run_line_magic('matplotlib', 'qt')
# plot_line(panda_df, parameters, metric='test_fde')
# plot_line_bar(panda_df, parameters, metric='test_fde')
plot_error_bar(panda_df, parameters, metric='min_test_fde')

