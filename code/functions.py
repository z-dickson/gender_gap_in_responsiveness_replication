import pandas as pd 
import numpy as np
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VECM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler










# create data 
def create_data(df, country):
    
    # create variables for the VAR model    
    df['issue_tweets'] = df.issue_tweets_male + df.issue_tweets_female
    df['total_tweets'] = df.total_tweets_male + df.total_tweets_female
    df['attention'] = df.issue_tweets / df.total_tweets

    # drop missing values (if any)
    df = df.loc[(df.mip_male != 0)&(df.mip_female != 0)]

    us = df[df['country'] == 'us']
    uk = df[df['country'] == 'uk']

    df=df.loc[df.issue != 'international_affairs']

    def lag_dv(df, col, lag):
        x = df.copy()
        x = x.sort_values(by=['date','issue'])
        x[f'{col}_{lag}'] = x.groupby(['issue'])[col].shift(-lag)
        return x 

    cols = ['mip_female', 'mip_male']


    for col in cols:
        us = lag_dv(us, col, 1)
        us = lag_dv(us, col, 2)
        uk = lag_dv(uk, col, 1)
        uk = lag_dv(uk, col, 2)
    












## create function to plot IRF with confidence intervals

def plot_irf(country, vars, cum=True): 
    
    if country == 'us':
        irf = us_irf
    elif country == 'uk':
        irf = uk_irf    
    
    if cum == True:
        values = irf.cum_effects
    else:
        values = irf.irfs
    
    stderr = irf.stderr()
    upper = values + 1.96*stderr
    lower = values - 1.96*stderr

    
    
    columns_dic = {'attention': 'All Reps\' Attention', 'mip_female_1': 'Women\'s Salience', 'mip_male_1': 'Men\'s Salience', 
                   'male_attention': 'Male Reps\' Attention', 'female_attention': 'Female Reps\' Attention', 'mip_female': 'Women\'s Salience', 'mip_male': 'Men\'s Salience', 
                   'mip_male_2': 'Men\'s Salience', 'mip_female_2': 'Women\'s Salience', 'male_minus_female': 'Male minus female salience', 
                   'issue_tweets_female': 'Women\'s Issue Tweets', 'issue_tweets_male': 'Men\'s Issue Tweets', 'issue_tweets': 'Issue Tweets', 'total_tweets': 'Total Tweets'}
    
    if country == 'us':
        names = us_results.names
    elif country == 'uk':
        names = uk_results.names
        
    ## granger causality tests for each variable combination         
        
    clean_names = pd.DataFrame(names)[0].map(columns_dic).to_list()
    
    
    
    titles = []

    for i in clean_names: 
        for j in clean_names:
            titles.append(j + '&#8658; <br> ' + i)
    
    # Create figure with subplots
    fig = make_subplots(rows=vars, cols=vars,
                       shared_xaxes=False,
                       shared_yaxes=False,
                       vertical_spacing=0.15, 
                       horizontal_spacing=0.05, 
                       subplot_titles=titles,
                       #specs=[[None, {}, {}],
                       #       [{}, None, {}],
                       #       [{}, {}, None]]
                       )
                       


    for i in range(vars):
        for j in range(vars):
            
         
            fig.add_trace(
                go.Scatter(
                    x = np.arange(len(values)),
                    y = values[:,i,j],
                    mode = 'lines+markers',
                    line=dict(width=4, color='red'),
                ), row=i+1, col=j+1
            )
         
            fig.add_trace(
               go.Scatter(
               x = np.arange(len(values)),
               y = upper[:,i,j],
               mode='lines',
               line=dict(width=1, dash='dash', color='black'), 
               showlegend=False
               ), row=i+1, col=j+1
        )
         
            fig.add_trace(
               go.Scatter(
               x = np.arange(len(values)),  
               y = lower[:,i,j],
               mode='lines', 
               fill='tonexty',
               line=dict(width=1, dash='dash', color='black'),  
               fillcolor='rgba(0,176,246,0.2)',
               showlegend=False
               ), row=i+1, col=j+1
        )
         # Get final y-value
            cumulative = np.sum(values[:, i, j])
            y_final_placement = np.max(values[:, i, j]) #+ upper[-1, i, j]
            mean = np.mean(values[:, i, j])
            
            ## get p-value of the cumulative effect
            upper_ci = upper[:, i, j]
            lower_ci = lower[:, i, j]

            
            
            ### get p-value from Granger Causality test
            
            if country == 'uk':
                r = uk_results.test_causality(names[i], [names[j]], kind='f').summary().as_html()
            elif country == 'us':
                r = us_results.test_causality(names[i], [names[j]], kind='f').summary().as_html()
            else:
                print('Please specify a country: us or uk')
            
            #print(r)
            gc_table = pd.DataFrame(pd.read_html(r, header=0, index_col=0)[0])#['p-value']
            gc_table.reset_index(inplace=True)
            #print(names[i] + ' -> ' + names[j])
            #print(gc_table)
            pval = 1
            pval = float(gc_table['p-value'].values[0])

                
                # get stars for p-value
                
            
            if pval < .05:
                stars = '&#128955;'
            else:
                stars = ''
         
            fig.add_annotation(
                text='&#931;' +': ' + str(round(cumulative, 2)) + f'{stars}' + '<br>' + '&mu;' + ': ' + str(round(mean, 2)) + f'{stars}',
                font=dict(color='black', size=20, family='Arial'),
                x=len(values)-1,
                y=y_final_placement,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                row=i+1,
                col=j+1
            )
            cumulative = np.mean(values[:, i, j])
         
            """fig.update_layout(
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 1, 2, 3, 4, 5],
                    ticktext = ['0', '1', '2', '3', '4', '5', '']
                    ))"""
         

         
    fig.update_layout(
        height=950 if vars == 4 else 700, width=950 if vars == 4 else 800,
        #title='Cumulative Effects', 
        showlegend=False,
        template='presentation',
        plot_bgcolor = "white",
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        title_font_size = 26,
        font_size = 18,
        legend_title_font_color="black", 
        xaxis_title="",
        title_x=0.05, 
        title_y=0.99,
        margin=dict(l=50, r=25, t=50, b=100, pad=1)
    )

    # add h line at 0
    fig.add_hline(y=0, line_width=1.5, line_dash="solid", line_color="black")
    fig.add_vline(x=0, line_width=1.5, line_dash="solid", line_color="black")
    # shade the background of the subplots 
    
    fig.update_annotations(font=dict(size=18, color="black", family='Arial'))
    
    fig.add_annotation(
        x=1.0,
        y=-.1,
        text=f"<i>Cumulative Effect Estimates and 95% Confidence Intervals.</i> VAR({irf.lags}) Model.<br>&#931; = Cumulative Effect; &mu; = Mean Effect; &#128955; = Granger causality (p-value < 0.05)",
        ## align text to left
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14, color="black", family='Arial'),
        opacity=0.8,
        )
    
    ## change x axis tick labels 
    fig.update_xaxes(tickfont=dict(size=14, color='black', family='Arial'))
    

    fig.show()
   
   
   
## plot IRF with confidence intervals using plotly 
def add_values(country, param, all_values=False, vars_containing=None):
    if country == 'us':
        results = us_results
        irf = us_irf
    elif country == 'uk':
        results = uk_results
        irf = uk_irf
    else:
        return 'Country not found'
    d = pd.DataFrame()
    
    if param == 'irf':
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                d[results.names[j] + '&#8658; <br> ' + results.names[i]] = irf.irfs[:,i,j]
    elif param == 'se':
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                d[results.names[j] + '&#8658; <br> ' + results.names[i]] = irf.cum_effect_stderr()[:,i,j]
    elif param == 'cum_effects':
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                d[results.names[j] + '&#8658; <br> ' + results.names[i]] = irf.cum_effects[:,i,j]
    elif param == 'names': 
        names = []
        for i in results.names:
            for j in results.names:
                if all_values == True: 
                    names.append(j + '&#8658; <br> ' + i)
                else:
                    if i != j:
                        names.append(j + '&#8658; <br> ' + i)
        if vars_containing == None:
            return names
        elif vars_containing != None: 
            names = pd.DataFrame(names)
            names = names.loc[names[0].str.contains(vars_containing, case=False, na=False)]    
            names = names[0].to_list()
            return names     
    elif param == 'gc':
        test_values = []
        idx_names = []
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                r = results.test_causality(results.names[i], [results.names[j]], kind='f').summary().as_html()   
                gc_table = pd.DataFrame(pd.read_html(r, header=0, index_col=0)[0])#['p-value']
                gc_table.reset_index(inplace=True)
                pval = 1
                pval = float(gc_table['p-value'].values[0])
                tval = float(gc_table['Test statistic'].values[0])
                test_values.append([pval, tval])
        x = pd.DataFrame(test_values, columns=['p-value', 't-value'])

        return x

    else:
        return 'Parameter not found'
    d = d.transpose()
    if all_values == True: 
        return d 
    elif all_values == False and vars_containing != None:
        return d.loc[d.index.str.contains(vars_containing)]
    else: 
        return d.loc[~d.index.isin(['male_attention->male_attention', 'female_attention->female_attention',
                    'mip_female->mip_female', 'mip_male->mip_male'])]










def plot_irf_v2(country, param='irf', all_values=False, vars_containing=None):
    
    names = add_values(country, 'names', all_values=all_values, vars_containing=None)
    names = [i.replace('->', ' -><br> ') for i in names]
    
    
    columns_dic = {'female_attention': 'Women Reps\' Attention','male_attention': 'Men Reps\' Attention', 'mip_female': 'Women\'s Salience', 'mip_male': 'Men\'s Salience', 
                   'issue_tweets_female': 'Women\'s Issue Tweets', 'issue_tweets_male': 'Men\'s Issue Tweets', 'issue_tweets': 'Issue Tweets', 'total_tweets': 'Total Tweets', '_1': ''}
    
    names = pd.DataFrame(names)[0].replace(columns_dic, regex=True).to_list()
    
    vars = 4
    
    fig = make_subplots(rows=2 if all_values==True else 3, cols=4,
                        shared_xaxes=False,
                        shared_yaxes=True,
                        vertical_spacing=0.15, 
                        horizontal_spacing=0.05, 
                        subplot_titles=names,
                        )


    ## plot irf values 
    vals = add_values(country, param, all_values=all_values, vars_containing=None)
    for N, i in enumerate(add_values(country, 'names', all_values=all_values, vars_containing=None)):  
        p = vals.loc[vals.index == i].values
        fig.add_trace(go.Scatter(x = np.arange(len(p[0])), y = p[0], mode = 'lines+markers', line=dict(width=2, color='red')), 
                      row= N // vars + 1, col = N % vars + 1)
        
        cumulative = np.sum(p[0])
        #y_final_placement = np.max(p[0])
        mean = np.mean(p[0])
        y_final_placement = np.max(p[0]) + np.mean(p[0])
        
        #print(vals.loc[vals.index == i].index[0].split('->'))
        x_val, y_val = vals.loc[vals.index == i].index[0].split('&#8658; <br> ')
        
        granger_cause = get_granger(country, x_val, y_val)
        ## get p-value of the cumulative effect
        print(granger_cause)
        
        if granger_cause[0] < .05:
            stars = '&#128955;'
        else:
            stars = ''
        
        
        fig.add_annotation(
            text='&#931;' +': ' + str(round(cumulative, 2)) + stars + '<br>' + '&mu;' + ': ' + str(round(mean, 2)) + stars,# + '<br>' + 't-val: ' + str(np.round(get_granger(country, x_val, y_val)[1],2)),
            font=dict(color='black', size=14, family='Arial'),
            x=len(p[0])-3.2,
            y=y_final_placement + .1,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            row=N // vars + 1,
            col=N % vars + 1
        )
    
    
    ## plot upper and lower confidence interval
    upper = vals + 1.96*add_values(country, 'se', all_values=all_values, vars_containing=None)
    lower = vals - 1.96*add_values(country, 'se', all_values=all_values, vars_containing=None)
    
    for N, i in enumerate(add_values(country, 'names', all_values=all_values, vars_containing=None)): 
        p_u = upper.loc[upper.index == i].values
        p_l = lower.loc[lower.index == i].values
        fig.add_trace(go.Scatter(x = np.arange(len(p_u[0])), y = p_u[0], mode='lines', line=dict(width=1, dash='dash', color='black'),
                                 showlegend=False), 
                      row= N // vars + 1, col = N % vars + 1)
        fig.add_trace(go.Scatter(x = np.arange(len(p_l[0])), y = p_l[0], mode='lines', line=dict(width=1, dash='dash', color='black'),
                                 showlegend=False,
                                 fill='tonexty',
               fillcolor='rgba(0,176,246,0.2)',), 
                      row= N // vars + 1, col = N % vars + 1)
        
        

    fig.add_annotation(
        x=1.0,
        y=-.1,
        text=f"<i>Cumulative Effect Estimates and 95% Confidence Intervals.</i> VAR(9) Model.<br>&#931; = Cumulative Effect; &mu; = Mean Effect; &#128955; = Granger causality (p-value < 0.05)",
        ## align text to left
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14, color="black", family='Arial'),
        opacity=0.8,
        )

    fig.update_layout(
        height=950, width=950,
        #title='Cumulative Effects' if param == 'cum_effects' else 'Impulse Response Functions', 
        showlegend=False,
        template='presentation',
        plot_bgcolor = "white",
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        title_font_size = 26,
        font_size = 18,
        legend_title_font_color="black", 
        xaxis_title="",
        title_x=0.05, 
        title_y=0.99,
        margin=dict(l=50, r=25, t=90, b=100, pad=1)
    )

    fig.show()



## helper function to assess granger causality

def get_granger(country, X, Y, kind='f', full = False):
    if country == 'uk':
        r = uk_results.test_causality(Y, [X], kind='f').summary().as_html()
    elif country == 'us':
        r = us_results.test_causality(Y, [X], kind='f').summary().as_html()
    else:
        print('Please specify a country: us or uk')
            
    gc_table = pd.DataFrame(pd.read_html(r, header=0, index_col=0)[0])
    gc_table.reset_index(inplace=True)
    if full == True:
        gc_table['X'] = X
        gc_table['Y'] = Y
        return gc_table

    pval = 1
    pval = float(gc_table['p-value'].values[0])
    tval = float(gc_table['Test statistic'].values[0])

            
    if pval < .05:
        stars = '&#128955;'
    else:
        stars = ''
         
    return pval, tval
