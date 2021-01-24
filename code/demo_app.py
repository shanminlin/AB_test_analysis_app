import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# Set layout and title
st.set_page_config(layout="wide")
st.title('A/B Test Report')

# Load data and formatting
df = pd.read_csv('../data/clean_data.csv')
df['event_time'] = pd.to_datetime(df['event_time'])
df['experiment_group'] = df['experiment_group'].map({'control': 'control', 
                                                     'experiment_1': 'variation 1', 
                                                     'experiment_2': 'variation 2',
                                                     'experiment_3': 'variation 3'})
df = df.rename(columns={'experiment_group': 'experiment'})
td = (df['event_time'].max() - df['event_time'].min()).days

# Overview in sidebar
st.sidebar.title('')
st.sidebar.title('')
st.sidebar.title('')
st.sidebar.title('')
st.sidebar.markdown(f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
  <div class="card text-white bg-info mb-2" style="width: 15rem">
    <div class="card-body">
      <h4 class="card-title"><center>Number of visitors</center></h4>
        <h4 class="card-text"><center>{df['user_uuid'].nunique():,d}</center></h4>
  </div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
  <div class="card text-white bg-secondary mb-2" style="width: 15rem">
    <div class="card-body">
      <h4 class="card-title"><center>Test duration (days)</center></h4>
        <h4 class="card-text"><center>{td:,d}</center></h4>
  </div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
  <div class="card text-white bg-success mb-2" style="width: 15rem">
    <div class="card-body">
      <h4 class="card-title"><center>Number of test variants</center></h4>
        <h4 class="card-text"><center>{df['experiment'].nunique()-1:,d}</center></h4>
  </div>
</div>""", unsafe_allow_html=True)

# helper functions
def click_through_probability_from_step_1_to_step_2(df):
    return df.loc[df['event_type'] == '#_of_users', 'user_uuid'].nunique() / df.loc[df['event_type'] == 'open', 'user_uuid'].nunique()

def click_through_probability_from_step_2_to_step_3(df):
    return df.loc[df['event_type'] == 'search', 'user_uuid'].nunique() / df.loc[df['event_type'] == '#_of_users', 'user_uuid'].nunique()

def click_through_probability_from_step_3_to_step_4(df):
    return df.loc[df['event_type'] == 'begin_ride', 'user_uuid'].nunique() / df.loc[df['event_type'] == 'search', 'user_uuid'].nunique()

def active_users(df):
    return df['user_uuid'].nunique()

def conversion_probability(df):
    return df.loc[df['event_type'] == 'begin_ride', 'user_uuid'].nunique() / df.loc[df['event_type'] == 'open', 'user_uuid'].nunique()

def compute_binomial_pvalue(n, success, p=0.5):
    success_range = range(success, n + 1)
    pvalue = stats.binom(n=n, p=p).pmf(k=success_range).sum()
    return pvalue

def two_proportions_ztest(a_conversion, a_size, b_conversion, b_size):
    a_prob = a_conversion / a_size
    b_prob = b_conversion / b_size
    total_size = a_size + b_size
    total_conversion = a_conversion + b_conversion
    pooled_prob = total_conversion / total_size
    
    var = pooled_prob * (1 - pooled_prob) * (1 / a_size + 1 / b_size)
    zscore = np.abs(a_prob - b_prob) / np.sqrt(var)
    one_side = 1 - stats.norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return pvalue

########################################################################
# Obvervations over time
# selections
col1, col2, col3 = st.beta_columns((8, 1, 8))
col1.subheader('Observations over time')
col3.subheader('Data breakdown')

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

selected_group = col3.selectbox('Group by', ('age', 'user neighborhood', 'experiment'), index=0)
selected_group_formatted = selected_group.replace(' ', '_')

selected_group_metric = col5.selectbox('Select a metric:', ('active users', 
                                                            'conversion probability',
                                                            'click through probability from step 1 to step 2',
                                                            'click through probability from step 2 to step 3',
                                                            'click through probability from step 3 to step 4'), key='1')
selected_group_metric_formatted = selected_group_metric.replace(' ', '_')
selected_group = col6.multiselect('Group by', ('age', 'user neighborhood', 'experiment'), default='age', key='1')

# compute
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
########################################################################
# Data breakdown
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
########################################################################
col1, col2, col3 = st.beta_columns((8, 1, 8))
col1.subheader('Significance test for conversions')
col3.subheader('Sign test for daily conversions')

# select paramters
col1, col2, col3, col4, col5 = st.beta_columns((4, 4, 1, 4, 4))
selected_test_significance = col1.selectbox('Select statistical test', ('two-tailed t test', 'two-tailed z test'), key='1')
selected_alpha_significance = col2.selectbox('Select alpha', ('0.05', '0.01'), key='1')

selected_test_sign = col4.selectbox('Select statistical test', ('two-tailed t test', 'two-tailed z test', 'binomial test'), index=1)
selected_alpha_sign = col5.selectbox('Select alpha', ('0.05', '0.01'), index=0)


col1, col2, col3 = st.beta_columns((8, 1, 8))
if selected_test_significance == 'two-tailed t test':
    col1.text('Sample size is large (n>50), use z test.')
else:
    ztest_result = pd.DataFrame()
    ztest_result['visits'] = df.groupby('experiment').apply(lambda x: x.loc[x['event_type'] == 'open', 'user_uuid'].nunique())
    ztest_result['conversions'] = df.groupby('experiment').apply(lambda x: x.loc[x['event_type'] == 'begin_ride', 'user_uuid'].nunique())
    ztest_result['conversion probability'] = ztest_result['conversions'] / ztest_result['visits']
    ztest_result['compare to control (%)'] = (ztest_result['conversion probability'] - ztest_result.loc['control', 'conversion probability']) / ztest_result.loc['control', 'conversion probability'] * 100
    a_size = ztest_result.loc['control', 'visits']
    a_conversion = ztest_result.loc['control', 'conversions']
    ztest_result['p-value'] = ztest_result.apply(lambda x: two_proportions_ztest(a_conversion, a_size, x['conversions'], x['visits']), axis=1)
    ztest_result['Significance level'] = [float(selected_alpha_significance)]*4
    ztest_result['Significance'] = ztest_result.apply(lambda x: x['p-value'] < x['Significance level'], axis=1)
    ztest_result['Achieved significance'] = ztest_result.apply(lambda x: 'Yes' if x['Significance']==1 else 'No', axis=1)
    ztest_result.drop(['Significance level', 'Significance'], axis=1, inplace=True)
    col1.table(ztest_result)

########################################################################
# Sign test for daily conversions
conversion_daily = df.groupby([pd.Grouper(freq='1D'), 'experiment']).apply(conversion_probability)
conversion_daily = conversion_daily.unstack().reset_index()
conversion_daily['var1_success'] = conversion_daily.apply(lambda x: 1 if x['variation 1'] > x['control'] else 0, axis=1)
conversion_daily['var2_success'] = conversion_daily.apply(lambda x: 1 if x['variation 2'] > x['control'] else 0, axis=1)
conversion_daily['var3_success'] = conversion_daily.apply(lambda x: 1 if x['variation 3'] > x['control'] else 0, axis=1)

n = len(conversion_daily)
p = 0.5 # null hypothesis

if selected_test_sign == 'two-tailed t test' or selected_test_sign == 'two-tailed z test':
    if n * p < 10 or n * (1 - p) < 10:
        col3.write('Sample size too small. Use binomial test.')
else:
    var1_success = sum(conversion_daily['var1_success'])
    var2_success = sum(conversion_daily['var2_success'])
    var3_success = sum(conversion_daily['var3_success'])   
    sign_test_result = pd.DataFrame(index=['variation 1', 'variation 2', 'variation 3'])
    sign_test_result['Total days'] = [n] * 3
    sign_test_result['Successes'] = [var1_success, var2_success, var3_success]
    sign_test_result['p-value'] = sign_test_result.apply(lambda x: compute_binomial_pvalue(x['Total days'],
                                                                                           x['Successes']),
                                                         axis=1
                                                        )
    sign_test_result['Significance level'] = [np.round(float(selected_alpha_sign), 2)]*3
    sign_test_result['Significance'] = sign_test_result.apply(lambda x: x['p-value'] < x['Significance level'], axis=1)
    sign_test_result['Achieved significance'] = sign_test_result.apply(lambda x: 'Yes' if x['Significance']==1 else 'No', axis=1)
    sign_test_result.drop(['Significance level', 'Significance'], axis=1, inplace=True)
    col3.table(sign_test_result)