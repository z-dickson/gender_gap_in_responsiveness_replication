import pandas as pd 
import numpy as np
import plotly.graph_objects as go 
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VECM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler







def print_MIP_descriptive_stats():

    mip = return_mip_data()
    mip['country'] = mip.country.str.upper()
    mip = mip.reset_index(drop=True)
    mip.date = mip.date.str.replace('.', '')
    mip.date = mip.date.str.slice(0, 10)
    mip['gender'] = mip.gender.replace({'Female': 'Women', 'Male': 'Men'})
    mip.issue = mip.issue.str.replace('_', ' ').str.title()
    mip.date = pd.to_datetime(mip.date)
    mip = mip.loc[(mip.date >= '2017-12-31')&(mip.date < '2022-01-01')]
    mip.columns = mip.columns.str.title()
    mip['Priority'] = mip.Priority.astype(float)


    x = mip.groupby(['Country','Issue','Gender']).Priority.describe().round(2).reset_index()

    ### formatting https://stackoverflow.com/questions/64499551/formatting-of-df-to-latex
    x[['count', 'mean', 'std', 'min', '25%',
        '50%', '75%', 'max']] = x[['count', 'mean', 'std', 'min', '25%',
        '50%', '75%', 'max']].applymap(lambda x: str.format("{:0_.2f}", x).replace('.', ',').replace('_', '.'))

    x=x.replace(',', '.', regex=True)

    print(x.to_latex(index=False))





def print_confusion_matrix(file):

    cm = pd.read_csv(f'../data/{file}')

    ## plot confusion matrix
    fig = px.imshow(cm, color_continuous_scale='Blues', 
                    labels=dict(x="Predicted", y="Actual", color="Frequency"), 
                    x=list(cm.columns), y=list(cm.columns), 
                    title='Confusion Matrix for UK Public Policy Priorities', 
                    width=800, height=800, color_continuous_midpoint=0.5, zmin=0, zmax=250, 
                    # add text for the values in each cell
                    text_auto=True
                    )


    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Arial",
        title_font_size=26,
        title_font_color="black",
        font_size = 12,
        legend_title_font_color="black", 
        template='plotly_white',
        title_text='Confusion Matrix: Model Predictions vs. Annotated Labels',
        #plot_bgcolor='#c1c8d4',
        #paper_bgcolor='#c1c8d4'
        )
    fig.show()









def mip_data(country, sheet_name):
    
    # this function gets the most important issues facing the country. It returns the data in a long format.
    
    try: 
        x = pd.read_excel(f'../data/{country}_MIP.xlsx', sheet_name=sheet_name)

    except ValueError:
        print('Sheet name not found. Please check the sheet name and try again. Names are `Male` and `Female`')
    
    x = x.transpose()
    x.columns = x.iloc[0]
    x = x.iloc[1:]
    x.columns = x.columns.to_list()
    x.columns = x.columns.str.replace(' ', '_').str.lower().str.replace("'", '').str.replace("/", '_').str.replace("&_", '')
    x['date'] = x.index
    if country == 'us':
        x['defense'] = x['terrorism'] + x['the_war_in_afghanistan'] + x.national_security_and_foreign_policy
        x['crime'] = x.gun_control + x.crime_and_criminal_justice_reform
        x['health'] = x.medicare + x['health_care']
        x['economy'] = x.jobs_and_the_economy + x.inflation_prices + x.inflation
        x['environment'] = x.climate_change_and_the_environment
        x['tax'] = x.taxes_and_government_spending
        

    if country == 'uk':
        x['international_affairs'] = x['britain_leaving_the_eu']
        x['immigration'] = x.immigration_asylum
        x['economy'] = x.the_economy
        x['environment'] = x.the_environment
        x['defense'] = x.defence_and_security + x.defence_and_terrorism + x.afghanistan
        
    x=x.melt(id_vars=['date'], value_vars=x.columns.to_list(), var_name='issue', value_name='priority')
    x['gender'] = sheet_name
    x['country'] = country
    issues = ['environment', 'defense', 'economy', 'health', 'tax',  'education', 'immigration', 'crime', 'international_affairs']
    x = x.loc[x.issue.isin(issues)]
    return x 




def return_mip_data():
    df = pd.DataFrame()
    for country in ['us', 'uk']:
        for gender in ['Male', 'Female']:
            x = mip_data(country, gender)
            df = pd.concat([df, x])
    df = df.reset_index(drop=True)
    df.date = df.date.str.replace('.', '')
    df.date = df.date.str.slice(0, 10)
    df['gender'] = df.gender.str.lower()
    return df




def make_mip_plot(mip, col_wrap, country):
    mip['gender'] = mip.gender.replace('male','men').replace('female','women')
    mip.date = pd.to_datetime(mip.date)
    mip = mip.loc[(mip.date >= '2017-12-25')&(mip.date < '2022-01-10')]
    mip.columns = mip.columns.str.title()

    mip['Gender'] = mip.Gender.str.title()
    mip['Issue'] = mip.Issue.str.replace('_', ' ').str.title()
    mip['Salience'] = mip.Priority
    
    ## make MIP figure for the UK
    colors = px.colors.qualitative.D3
    x = mip.loc[mip.Country == country]

    fig = px.scatter(x, x='Date', y='Salience', color='Gender', 
                    trendline="lowess",
                    trendline_options=dict(frac=0.1),
                    opacity=0.2, 
                    #trendline_options=dict(function="mean", window=12),
                    facet_col='Issue', facet_col_wrap=col_wrap, color_discrete_sequence=[colors[0], colors[1]], 
                    facet_col_spacing=0.1
                    )

    fig.update_layout(
                title_font_size=26,
                font_family="Arial",
                font_color="black",
                title_font_family="Arial",
                title_font_color="black",
                font_size = 15,
                legend_title_font_color="black", 
                template='plotly_white',
                #title = 'UK Public Policy Priorities',
                height=800, width=900,
                )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        title = 'Gender',
        font=dict(
            size=16,), 
        x=0.3
    ))


    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    tr_line=[]
    for  k, trace  in enumerate(fig.data):
            if trace.mode is not None and trace.mode == 'lines':
                tr_line.append(k)

    for id in tr_line:
        fig.data[id].update(line_width=3)

    ## make x axis labels horizontal
    fig.update_xaxes(tickangle=45)

    fig.update_layout(legend= {'itemsizing': 'constant'})

    ## make y axis labels a percentage with 2 decimal places
    fig.update_yaxes(tickformat=',.0%')#,range= [0,1])
    fig.update_yaxes(matches=None)
    # add y axis ticks to each facet
    fig.update_yaxes(showticklabels=True)
    # add space between facets
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        x=0.4,
        xanchor="right"),
        font=dict( 
            family="Arial", 
            size=18, 
            color="black"
        ),
        #marker_size=15
        )

    fig.show()






def make_attention_plot(country, lowess_frac):
    
    
    path = '../data/attention_time_series_v2.csv'

    df = pd.read_csv(path)

    df['issue_tweets'] = df.issue_tweets_male + df.issue_tweets_female
    df['total_tweets'] = df.total_tweets_male + df.total_tweets_female
    df['attention'] = df.issue_tweets / df.total_tweets
    df['issue'] = df['issue'].str.replace('_', ' ').str.title()
    df.loc[(df.country == 'uk') & (df.issue == 'International Affairs'), 'issue'] = 'Brexit'
    

    ## make MIP figure for the UK
    
    
    colors = px.colors.qualitative.D3
    
    
    if country == 'uk':
        x = df.loc[df.country == 'uk']
    elif country == 'us':
        x = df.loc[df.country == 'us']
    else: 
        raise ValueError('Country not found. Please enter `uk` or `us`')

    fig = make_subplots(rows=3, cols=3,
                       shared_xaxes=False,
                       shared_yaxes=False,
                       vertical_spacing=0.08, 
                       horizontal_spacing=0.1, 
                       subplot_titles=x.issue.unique(),
                       )

    def replace_outliers_with_nan(vector):
            mean_value = np.mean(vector)
            std_dev = np.std(vector)
            # Create a mask for values outside 3 standard deviations
            mask = np.abs(vector - mean_value) > 3 * std_dev
            # Replace outlier values with NaN
            vector[mask] = np.nan
            return vector
    
    #fig = go.Figure()
    for i, issue in enumerate(x.issue.unique()):
        data = x.loc[x.issue == issue]
        
        
        data['lowess_men'] = sm.nonparametric.lowess(data.male_attention, pd.Categorical(data.date).codes, frac=lowess_frac)[:, 1]
        fig.add_trace(go.Scatter(x=data.date, y=data.lowess_men, mode='lines', name = 'Men\'s Attention', line=dict(color=colors[0], width=3),
                                 showlegend=True if i == 0 else False), 
                      row=i // 3 + 1, col=i % 3 + 1)
        
        data['lowess_women'] = sm.nonparametric.lowess(data.female_attention, pd.Categorical(data.date).codes, frac=lowess_frac)[:, 1]
        fig.add_trace(go.Scatter(x=data.date, y=data.lowess_women, mode='lines', name='Women\'s Attention', line=dict(color=colors[1], width=3), showlegend=True if i == 0 else False),
                      row=i // 3 + 1, col=i % 3 + 1)
        
        
        ## remove all values over 3 standard deviations from the mean so the lowess line is visible while including the points. This is done !!!!AFTER!!!! the lowess line is calculated
        
        

        data.female_attention = replace_outliers_with_nan(data.female_attention.values)
        data.male_attention = replace_outliers_with_nan(data.male_attention.values)
        
        fig.add_trace(go.Scatter(x=data.date, y=data.male_attention, mode='markers', name='Men',opacity=.15, line=dict(color=colors[0]), 
                                 showlegend=False), 
                      row=i // 3 + 1, col=i % 3 + 1,)
        fig.add_trace(go.Scatter(x=data.date, y=data.female_attention, mode='markers', name='Women',opacity=.15, line=dict(color=colors[1]),
                                 showlegend=False),
                      row=i // 3 + 1, col=i % 3 + 1)
        

    fig.update_layout(
                title_font_size=26,
                font_family="Arial",
                font_color="black",
                title_font_family="Arial",
                title_font_color="black",
                font_size = 15,
                legend_title_font_color="black", 
                template='plotly_white',
                #title = 'UK Public Policy Priorities',
                height=1000, width=900,
                showlegend=True)
    fig.update_yaxes(tickformat=',.0%')#,range= [0,1])
    fig.update_yaxes(matches=None)
    # add y axis ticks to each facet
    fig.update_yaxes(showticklabels=True)
    # update margins 
    
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    
    # make legend horizontal
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        x=0.4,
        xanchor="right"),
        font=dict( 
            family="Arial", 
            size=18, 
            color="black"
        ),
        #marker_size=15
        )
    fig.update_layout(legend= {'itemsizing': 'constant'})
    return fig.show()







 



# create data 
def create_TS_data(path, country):
    
    # read data
    df = pd.read_csv(path)

    # set index to be date 
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # create attention variables (attention = issue tweets / total tweets)
    df['male_attention'] = df['issue_tweets_male']/df['total_tweets_male']
    df['female_attention'] = df['issue_tweets_female']/df['total_tweets_female']
    
    # create datasets for the country 
    ts = df[df['country'] == country]
    
    # remove columns that are not needed for the analysis
    ts = ts[['male_attention', 'female_attention', 'mip_female', 'mip_male']]
    
    # replace 0 values with np.nan 
    ts = ts.replace(0, np.nan).dropna()
    
    # get the log ratio of all variables
    ts = np.log(ts.div(ts.sum(axis=1), axis=0)).dropna()
    
    return ts 
    
    


    
    
# create VAR model

def create_VAR_model(data, lags, irf_horizon):
    model = VAR(data)
    results = model.fit(lags, ic='aic')
    irf = results.irf(irf_horizon)
    return results, irf



    
    
def add_values(results, irf_model, estimate, all_values=False, vars_containing=None):

    d = pd.DataFrame()
    
    if estimate == 'irf':
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                d[results.names[j] + '&#8658; <br> ' + results.names[i]] = irf_model.irfs[:,i,j]
    elif estimate == 'se':
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                d[results.names[j] + '&#8658; <br> ' + results.names[i]] = irf_model.cum_effect_stderr()[:,i,j]
    elif estimate == 'cumulative_effects':
        for i in range(0, len(results.names)):
            for j in range(0, len(results.names)):
                d[results.names[j] + '&#8658; <br> ' + results.names[i]] = irf_model.cum_effects[:,i,j]
    elif estimate == 'names': 
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
    elif estimate == 'gc':
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





def plot_irf(results, irf_model, estimate, all_values=False, vars_containing=None):
    
    names = add_values(results, irf_model, 'names', all_values=all_values, vars_containing=None)
    names = [i.replace('->', ' -><br> ') for i in names]
    #print(names)
    
    
    columns_dic = {'female_attention': 'Women Reps\' Attention','male_attention': 'Men Reps\' Attention', 'mip_female': 'Women\'s Salience', 'mip_male': 'Men\'s Salience', 
                   'issue_tweets_female': 'Women\'s Issue Tweets', 'issue_tweets_male': 'Men\'s Issue Tweets', 'issue_tweets': 'Issue Tweets', 'total_tweets': 'Total Tweets'}
    
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
    vals = add_values(results, irf_model, estimate, all_values=all_values, vars_containing=None)
    for N, i in enumerate(add_values(results, irf_model, 'names', all_values=all_values, vars_containing=None)):  
        p = vals.loc[vals.index == i].values
        fig.add_trace(go.Scatter(x = np.arange(len(p[0])), y = p[0], mode = 'lines+markers', line=dict(width=2, color='red')), 
                      row= N // vars + 1, col = N % vars + 1)
        
        cumulative = np.sum(p[0])
        #y_final_placement = np.max(p[0])
        mean = np.mean(p[0])
        y_final_placement = np.max(p[0]) + np.mean(p[0])
        
        #print(vals.loc[vals.index == i].index[0].split('->'))
        x_val, y_val = vals.loc[vals.index == i].index[0].split('&#8658; <br> ')
        
        granger_cause = get_granger(results, x_val, y_val)
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
    upper = vals + 1.96*add_values(results, irf_model, 'se', all_values=all_values, vars_containing=None)
    lower = vals - 1.96*add_values(results, irf_model, 'se', all_values=all_values, vars_containing=None)
    
    for N, i in enumerate(add_values(results, irf_model, 'names', all_values=all_values, vars_containing=None)): 
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

def get_granger(results, X, Y, kind='f', full = False):

    r = results.test_causality(Y, [X], kind='f').summary().as_html()
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







# granger table in appendix 

def granger_table(results):
    tab = pd.DataFrame()
    for i in ['male_attention', 'female_attention', 'mip_female', 'mip_male']:
        for j in ['male_attention', 'female_attention', 'mip_female', 'mip_male']:
            #print(i, j)
            tab = pd.concat([tab, get_granger(results, i, j, full=True)])
            
    tab['X'], tab['Y'] = tab['X'].str.replace('female_attention','Women Reps\' Attention').str.replace('male_attention', 'Men Reps\' Attention').str.replace('mip_female', 'Women\'s Salience').str.replace('mip_male', 'Men\'s Salience'), tab['Y'].str.replace('female_attention','Women Reps\' Attention').str.replace('male_attention', 'Men Reps\' Attention').str.replace('mip_female', 'Women\'s Salience').str.replace('mip_male', 'Men\'s Salience')
    
    tab['Coefficient'] = tab['X'] + ' -> ' + tab['Y']
    tab = tab[['Coefficient', 'Test statistic', 'p-value', 'Critical value', 'df']]
    
    print(tab.to_latex(index=False))
    
    
    
    
