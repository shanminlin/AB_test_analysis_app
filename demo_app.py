import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

@st.cache
def load_data():
    """Loads and formats data and formats."""
    df = pd.read_csv('data/clean_data.csv')
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['experiment_group'] = df['experiment_group'].map({'control': 'control', 
                                                         'experiment_1': 'variation 1', 
                                                         'experiment_2': 'variation 2',
                                                         'experiment_3': 'variation 3'})
    df = df.rename(columns={'experiment_group': 'experiment'})
    return df

def compute_duration(df):
    duration = (df['event_time'].max() - df['event_time'].min()).days
    return duration

def create_sidebars(df):
    """Creates sidebars for overview of the experiment."""
    # some empty space for a better look
    st.sidebar.title('')
    st.sidebar.title('')
    st.sidebar.title('')
    st.sidebar.title('')
    
    # format sidebar for total visits
    st.sidebar.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
      <div class="card text-white bg-info mb-2" style="width: 15rem">
        <div class="card-body">
          <h4 class="card-title"><center>Number of visitors</center></h4>
            <h4 class="card-text"><center>{df['user_uuid'].nunique():,d}</center></h4>
      </div>
    </div>""", unsafe_allow_html=True)
    
    # format sidebar for duration
    duration = compute_duration(df)
    st.sidebar.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
      <div class="card text-white bg-secondary mb-2" style="width: 15rem">
        <div class="card-body">
          <h4 class="card-title"><center>Test duration (days)</center></h4>
            <h4 class="card-text"><center>{duration:,d}</center></h4>
      </div>
    </div>""", unsafe_allow_html=True)
    
    # format sidebar for number of variations
    st.sidebar.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
      <div class="card text-white bg-success mb-2" style="width: 15rem">
        <div class="card-body">
          <h4 class="card-title"><center>Number of test variants</center></h4>
            <h4 class="card-text"><center>{df['experiment'].nunique()-1:,d}</center></h4>
      </div>
    </div>""", unsafe_allow_html=True)

def click_through_probability_from_step_1_to_step_2(df):
    opens = df.loc[df['event_type'] == 'open', 'user_uuid'].nunique()
    num_of_users = df.loc[df['event_type'] == '#_of_users', 'user_uuid'].nunique()
    return num_of_users / opens

def click_through_probability_from_step_2_to_step_3(df):
    num_of_users = df.loc[df['event_type'] == '#_of_users', 'user_uuid'].nunique()
    searches = df.loc[df['event_type'] == 'search', 'user_uuid'].nunique()
    return searches / num_of_users

def click_through_probability_from_step_3_to_step_4(df):
    searches = df.loc[df['event_type'] == 'search', 'user_uuid'].nunique()
    rides = df.loc[df['event_type'] == 'begin_ride', 'user_uuid'].nunique()    
    return rides / searches

def active_users(df):
    return df['user_uuid'].nunique()

def conversion_probability(df):
    opens = df.loc[df['event_type'] == 'open', 'user_uuid'].nunique()
    rides = df.loc[df['event_type'] == 'begin_ride', 'user_uuid'].nunique()
    return rides / opens

def compute_binomial_pvalue(n, success, p=0.5):
    """Conducts binomial test for number of successes.
    
    Parameters
    ----------
    n : int
        Number of observations
        
    success : int
        Number of successes
        
    p : float
        Probability of successes in null hypothesis
    
    Returns
    -------
    pvalue : float
        p-value for the binomial test
    """
    success_range = range(success, n + 1)
    pvalue = stats.binom(n=n, p=p).pmf(k=success_range).sum()
    return pvalue

def two_proportions_ztest(a_conversion, a_size, b_conversion, b_size):
    """Conducts z test for differences between two proportions.
    
    Parameters
    ----------
    a_conversion : int
        Number of conversions in sample a
        
    b_conversion : int
        Number of conversions in sample b
        
    a_size : int
        Number of observations in sample a
    
    b_size : int
        Number of observations in sample b
    
    Returns
    -------
    pvalue : float
        p-value for the z-test
    """
    a_prob = a_conversion / a_size
    b_prob = b_conversion / b_size
    total_size = a_size + b_size
    total_conversion = a_conversion + b_conversion
    pooled_prob = total_conversion / total_size
    
    # compute pooled variance assuming sample a and b having the same variance
    var = pooled_prob * (1 - pooled_prob) * (1 / a_size + 1 / b_size)
    
    zscore = np.abs(a_prob - b_prob) / np.sqrt(var)
    one_side = 1 - stats.norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return pvalue

def two_proportions_conf_interval(a_conversion, a_size, b_conversion, b_size, significance):
    """Computes the confidence interval for differences between two proportions.
    
    Parameters
    ----------
    a_conversion : int
        Number of conversions in sample a
        
    b_conversion : int
        Number of conversions in sample b
        
    a_size : int
        Number of observations in sample a
    
    b_size : int
        Number of observations in sample b
    
    significance: float
        alpha for significance test
        
    Returns
    -------
    lower_ci : float
        lower bound of the confidence interval
        
    upper_ci : float
        upper bound of the confidence interval
    """
    a_prob = a_conversion / a_size
    b_prob = b_conversion / b_size
    
    variance = a_prob * (1 - a_prob) / a_size + b_prob * (1 - b_prob) / b_size # unpooled
    sd = np.sqrt(variance)
    
    confidence = 1 - significance
    zcritical = stats.norm(loc=0, scale=1).ppf(confidence + significance/2)

    diff = abs(a_prob - b_prob)
    lower_ci = diff - zcritical * sd
    upper_ci = diff + zcritical * sd

    return lower_ci, upper_ci

def compute_and_plot(df):
    """
    Computes metrics over time, breaks down data into groups and plot.
    
    Parameters
    ----------
    df : Pandas dataframe
        Recorded event data from the experiment
    """
    # format GUI
    col1, col2, col3 = st.beta_columns((8, 1, 8))
    col1.subheader('Observations over time')
    col3.subheader('Data breakdown')

    # select box for metrics over time
    col1, col2, col3, col4, col5, col6 = st.beta_columns((4, 2, 2, 1, 4, 4))
    selected_daily_metric = col1.selectbox('Select a metric:', ('active users', 
                                                              'conversion probability',
                                                              'click through probability from step 1 to step 2',
                                                              'click through probability from step 2 to step 3',
                                                              'click through probability from step 3 to step 4'))
    selected_daily_metric_formatted = selected_daily_metric.replace(' ', '_')

    selected_frequency = col2.selectbox('Select frequency', ('hourly', 'daily'), index=1)
    selected_frequency_mapper = {'hourly': '1H', 'daily': '1D'}
    selected_frequency_formatted = selected_frequency_mapper[selected_frequency]

    # select box for groups over time
    selected_group = col3.selectbox('Group by', ('age', 'user neighborhood', 'experiment'), index=0)
    selected_group_formatted = selected_group.replace(' ', '_')

    # select box for metrics for data breakdown
    selected_group_metric = col5.selectbox('Select a metric:', ('active users', 
                                                                'conversion probability',
                                                                'click through probability from step 1 to step 2',
                                                                'click through probability from step 2 to step 3',
                                                                'click through probability from step 3 to step 4'), key='1')
    selected_group_metric_formatted = selected_group_metric.replace(' ', '_')
    
    # select box for groups for data breakdown
    selected_group = col6.multiselect('Group by', ('age', 'user neighborhood', 'experiment'), default='age', key='1')

    # compute metrics over time
    df = df.set_index('event_time')
    metric_daily = df.groupby([pd.Grouper(freq=selected_frequency_formatted), selected_group_formatted]).apply(eval(selected_daily_metric_formatted))
    metric_daily = metric_daily.unstack().reset_index()

    # plot
    fig_daily = px.line(metric_daily, x='event_time', y=df[selected_group_formatted].unique())

    fig_daily.update_xaxes(rangeslider_visible=True,
                           rangeselector=dict(buttons=list([dict(count=1, label="1d", step="day", stepmode="backward"),
                                                            dict(count=7, label="1w", step="day", stepmode="todate"),
                                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                                            dict(step="all")])))
    fig_daily.update_layout(title=selected_daily_metric,
                            title_x=0.5,
                            xaxis_title="Event time",
                            yaxis_title=selected_daily_metric)
    fig_daily.update_layout(width=750, height=500)

    col1, col2, col3 = st.beta_columns((8, 1, 8))
    col1.plotly_chart(fig_daily)

    
    # compute and plot for data breakdown
    if len(selected_group) == 1:
        selected_group1 = selected_group[0]
        selected_group1_formatted = selected_group1.replace(' ', '_')
        group1 = pd.DataFrame(df.groupby(selected_group1_formatted).apply(eval(selected_group_metric_formatted))).reset_index()
        group1.columns = [selected_group1, selected_group_metric]
        
        # plot
        fig_group = px.bar(group1, x=selected_group1, y=selected_group_metric)
        fig_group.update_layout(title='Distribution of {}<br>in each {} group'.format(selected_group_metric, selected_group1), title_x=0.5)
        fig_group.update_layout(width=700, height=500, bargap=0.2)
        col3.plotly_chart(fig_group)

    elif len(selected_group) == 2:
        selected_group1 = selected_group[0]
        selected_group1_formatted = selected_group1.replace(' ', '_')
        selected_group2 = selected_group[1]
        selected_group2_formatted = selected_group2.replace(' ', '_')
        group2 = pd.DataFrame(df.groupby([selected_group1_formatted, selected_group2_formatted]).apply(eval(selected_group_metric_formatted))).reset_index()
        group2.columns = [selected_group1, selected_group2, selected_group_metric]
        
        # plot
        fig_group = px.bar(group2, x=selected_group1, y=selected_group_metric, color=selected_group2, barmode = 'stack')
        fig_group.update_layout(title='Distribution of {} and {}<br>in each {} group'.format(selected_group_metric, selected_group2, selected_group1), title_x=0.5)
        fig_group.update_layout(width=700, height=500)
        col3.plotly_chart(fig_group)

        
def compute_statistics_two_proportions(df):
    """
    Computes statistics for conversion changes.
    
    Parameters
    ----------
    df : Pandas dataframe
        Recorded event data from the experiment
    """
    # format GUI
    col1, col2, col3 = st.beta_columns((8, 1, 8))
    col1.subheader('Significance test for conversions')

    # select paramters for statistical test for conversion changes
    col1, col2, col3, col4, col5 = st.beta_columns((4, 4, 1, 4, 4))
    test_significance = col1.selectbox('Select statistical test', ('two-tailed t test', 'two-tailed z test'), key='1')
    alpha_significance = col2.selectbox('Select alpha', ('0.05', '0.01'), key='1')
    alpha_significance = float(alpha_significance)
    
    # compute statistics for conversion changes    
    if test_significance == 'two-tailed t test':
        st.text('Sample size is large (n>50), use z test.')
        
    else:
        ztest_result = pd.DataFrame()
        ztest_result['visits'] = df.groupby('experiment').apply(lambda x: x.loc[x['event_type'] == 'open', 'user_uuid'].nunique())
        ztest_result['conversions'] = df.groupby('experiment').apply(lambda x: x.loc[x['event_type'] == 'begin_ride', 'user_uuid'].nunique())
        ztest_result['conversion probability'] = ztest_result['conversions'] / ztest_result['visits']
        ztest_result['compare to control (%)'] = (ztest_result['conversion probability'] - ztest_result.loc['control', 'conversion probability']) / ztest_result.loc['control', 'conversion probability'] * 100
        a_size = ztest_result.loc['control', 'visits']
        a_conversion = ztest_result.loc['control', 'conversions']
        
        # conduct z test
        ztest_result['lower bound'], ztest_result['upper bound'] = ztest_result.apply(lambda x: 
                                                                                      two_proportions_conf_interval(a_conversion,
                                                                                                                    a_size,
                                                                                                                    x['conversions'],
                                                                                                                    x['visits'],
                                                                                                                    alpha_significance), 
                                                                                      axis=1,
                                                                                      result_type='expand').T.values
       
        ztest_result['lower bound (%)'] = ztest_result.apply(lambda x: x['lower bound'] / ztest_result.loc['control', 'conversion probability'] * 100, axis=1)
        ztest_result['upper bound (%)'] = ztest_result.apply(lambda x: x['upper bound'] / ztest_result.loc['control', 'conversion probability'] * 100, axis=1)
        ztest_result['p-value'] = ztest_result.apply(lambda x: two_proportions_ztest(a_conversion, a_size, x['conversions'], x['visits']), axis=1)
        ztest_result['Significance level'] = [alpha_significance]*4
        ztest_result['Significance'] = ztest_result.apply(lambda x: x['p-value'] < x['Significance level'], axis=1)
        ztest_result['Achieved significance'] = ztest_result.apply(lambda x: 'Yes' if x['Significance']==1 else 'No', axis=1)
        ztest_result.drop(['Significance level', 'Significance', 'lower bound', 'upper bound'], axis=1, inplace=True)
        
        # show result
        st.table(ztest_result)
        
    
def compute_sign_test(df):
    """conducts sign test on day-by-day breakdown data."""

    st.subheader('Sign test for daily conversions')
    
    # select paramters for statistical test for conversion changes
    col1, col2, col3, col4, col5 = st.beta_columns((4, 4, 1, 4, 4))
    test_sign = col1.selectbox('Select statistical test', ('two-tailed t test', 'two-tailed z test', 'binomial test'), index=1)
    alpha_sign = col2.selectbox('Select alpha', ('0.05', '0.01'), index=0)
    alpha_sign = float(alpha_sign)
    
    # compute statistics for sign test for daily conversions
    df = df.set_index('event_time')
    conversion_daily = df.groupby([pd.Grouper(freq='1D'), 'experiment']).apply(conversion_probability)
    conversion_daily = conversion_daily.unstack().reset_index()
    conversion_daily['var1_success'] = conversion_daily.apply(lambda x: 1 if x['variation 1'] > x['control'] else 0, axis=1)
    conversion_daily['var2_success'] = conversion_daily.apply(lambda x: 1 if x['variation 2'] > x['control'] else 0, axis=1)
    conversion_daily['var3_success'] = conversion_daily.apply(lambda x: 1 if x['variation 3'] > x['control'] else 0, axis=1)

    n = len(conversion_daily)
    p = 0.5 # null hypothesis
    
    col1, col2, col3 = st.beta_columns((8, 1, 8))
    if test_sign == 'two-tailed t test' or test_sign == 'two-tailed z test':
        if n * p < 10 or n * (1 - p) < 10:
            col1.write('Sample size too small. Use binomial test.')
    else:
        var1_success = sum(conversion_daily['var1_success'])
        var2_success = sum(conversion_daily['var2_success'])
        var3_success = sum(conversion_daily['var3_success'])   
        sign_test_result = pd.DataFrame(index=['variation 1', 'variation 2', 'variation 3'])
        sign_test_result['Total days'] = [n] * 3
        sign_test_result['Successes'] = [var1_success, var2_success, var3_success]
        
        # conduct binomial test
        sign_test_result['p-value'] = sign_test_result.apply(lambda x: compute_binomial_pvalue(x['Total days'], x['Successes']), axis=1)
        sign_test_result['Significance level'] = [alpha_sign]*3
        sign_test_result['Significance'] = sign_test_result.apply(lambda x: x['p-value'] < x['Significance level'], axis=1)
        sign_test_result['Achieved significance'] = sign_test_result.apply(lambda x: 'Yes' if x['Significance']==1 else 'No', axis=1)
        sign_test_result.drop(['Significance level', 'Significance'], axis=1, inplace=True)
        
        # show result
        col1.table(sign_test_result)
        
def main():
    st.set_page_config(layout="wide")
    st.title('A/B Test Report')
    df = load_data()     
    create_sidebars(df)
    compute_and_plot(df)
    compute_statistics_two_proportions(df)
    compute_sign_test(df)
    
#------------------------------------------------------#

if __name__ == '__main__':
    main()
    