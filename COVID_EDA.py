#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for data preprocessing
import pandas as pd

# for conducting ANOVA
import numpy as np
import random
import scipy.stats as stats

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ### DATA CLEANING

# Confirmed cases:

# In[ ]:


df1=pd.read_csv(r"https://github.com/jgehrcke/covid-19-germany-gae/blob/master/cases-rki-by-state.csv?raw=true")
df1=df1.iloc[:383] # the data set gets updated daily. Thus we need to limit the entries in dataframe to derive meaningful conclusions
total_confirmed_cases=df1.iloc[382]['sum_cases'] # total confirmed cases as on 19.03.2021
df1 # cumulative dataset


# In[ ]:


# creating a dataframe for total confirmed cases in each state
state_data = pd.DataFrame(df1.loc[382])
state_data.drop(['time_iso8601','sum_cases'],inplace = True)
state_data.columns = ['Total Confirmed Cases']
state_data=state_data.rename_axis('State code').reset_index()
state_data


# In[ ]:


df1.info() # checking column data types and missing values


# In[ ]:


# cleaning dataset
date_only=pd.to_datetime(df1['time_iso8601']).dt.date # extracting date from date time object
df1['time_iso8601']=pd.to_datetime(date_only) # assigning it to column in dataframe
df1.rename(columns={'time_iso8601':'Date'},inplace=True) # renaming columns

# columns with numerical values
num_columns=['DE-BB', 'DE-BE', 'DE-BW', 'DE-BY', 'DE-HB', 'DE-HE', 'DE-HH',
       'DE-MV', 'DE-NI', 'DE-NW', 'DE-RP', 'DE-SH', 'DE-SL', 'DE-SN', 'DE-ST',
       'DE-TH', 'sum_cases']

i=382
while i>0 :
  df1.loc[i,num_columns]=df1.loc[i,num_columns]-df1.loc[i-1,num_columns]
  i-=1
df1


# Death cases:

# In[ ]:


df_deaths=pd.read_csv(r"https://github.com/jgehrcke/covid-19-germany-gae/blob/master/deaths-rki-by-state.csv?raw=true")
df_deaths=df_deaths.iloc[:383] # the data set gets updated daily. Thus we need to limit the entries in dataframe to derive meaningful conclusions
total_death_cases=df_deaths.iloc[382]['sum_deaths']
df_deaths.tail()


# In[ ]:


# total death cases for each state
dummy = pd.DataFrame(df_deaths.loc[382]).drop(['time_iso8601','sum_deaths'])
state_data['Total Death Cases']=list(dummy[382])
state_data


# In[ ]:


df_deaths.info()


# In[ ]:


# columns with numerical values
num_columns_death=['DE-BB', 'DE-BE', 'DE-BW', 'DE-BY', 'DE-HB', 'DE-HE', 'DE-HH',
       'DE-MV', 'DE-NI', 'DE-NW', 'DE-RP', 'DE-SH', 'DE-SL', 'DE-SN', 'DE-ST',
       'DE-TH', 'sum_deaths']

date_only=pd.to_datetime(df_deaths['time_iso8601']).dt.date
df_deaths['time_iso8601']=pd.to_datetime(date_only) # assigning it to column in dataframe
df_deaths.rename(columns={'time_iso8601':'Date'},inplace=True) 

i=df_deaths.shape[0]-1
while i>0 :
    df_deaths.loc[i,num_columns_death]=df_deaths.loc[i,num_columns_death]-df_deaths.loc[i-1,num_columns_death]
    i-=1
    
df_deaths


# Both confirmed and death cases:

# In[ ]:


# dataframe containing daily confirmed cases and death cases
total_data=pd.concat([df1[['Date','sum_cases']],df_deaths[['sum_deaths']]],axis=1)
total_data


# A dictionary to store state names and their codes:

# In[ ]:


# dictionary containing state codes in columns as keys and their actual names
list_of_states={ 'DE-BW':'Baden-Württemberg','DE-BY':'Bavaria','DE-BE':'Berlin','DE-BB':'Brandenburg',
                 'DE-HB':'Bremen','DE-HH':'Hamburg','DE-HE':'Hesse','DE-NI':'Lower Saxony',
                 'DE-MV':'Mecklenburg-Vorpommern','DE-NW':'North Rhine-Westphalia','DE-RP':'Rhineland-Palatinate',
                 'DE-SL':'Saarland','DE-SN':'Saxony','DE-ST':'Saxony-Anhalt','DE-SH':'Schleswig-Holstein',
                 'DE-TH':'Thuringia'
                }


# ### DATA VISUALIZATION ON COUNTRY DATA

# In[ ]:


fig=px.line(total_data,x='Date',y='sum_cases',title='Daily confirmed cases')
fig.show()


# ### MORTALITY RATE FOR COVID 19

# In[ ]:


# mortality rate from our dataset
MR=total_death_cases*100/total_confirmed_cases
print('Mortality rate from our dataset: ',MR)


# In[ ]:


# mortality rate from WHO data as on 19.03.2021
confirmed_cases_WHO=2645783
death_cases_WHO=74565
MR_WHO=death_cases_WHO*100/confirmed_cases_WHO
print("Mortality rate from W.H.O for Germany",MR_WHO)


# We can see that the two mortality rates are approximately similar 

# ### EXPLORATORY DATA ANALYSIS

# 1. Contribution of each state to covid cases in Germany

# In[ ]:


plt.pie(state_data['Total Confirmed Cases'],labels=state_data['State code'])
plt.gcf().set_size_inches(13, 9)


# The state of North Rhine-Westphalia (DE-NW) contributes the most to total covid cases in the country. This might be because North Rhine-Westphalia is the most [populated](https://en.wikipedia.org/wiki/List_of_German_states_by_population#:~:text=List%20of%20German%20states%20by%20population%20%20,%20%202,578,312%20%2013%20more%20rows) state in Germany. The state of Bremen (DE-HB) is the least populated and has the least covid cases among all states

# 2. Compare mortality rate for each state
# 
# 
# 
# 
# 

# In[ ]:


state_data['Mortality Rate']=state_data['Total Death Cases']*100/state_data['Total Confirmed Cases']
state_data


# In[ ]:


# find the state with highest mortality rate
state_data['Mortality Rate']=pd.to_numeric(state_data['Mortality Rate'])
max_index=state_data['Mortality Rate'].idxmax(axis=0)
print('The state with highest mortality rate is: ',list_of_states[state_data.iloc[max_index,0]])


# In[ ]:


# find the state with lowest mortality rate
min_index=state_data['Mortality Rate'].idxmin(axis=0)
print('The state with lowest mortality rate is: ',list_of_states[state_data.iloc[min_index,0]])


# In[ ]:


# plotting mortality rate for different states
state_data.plot.bar(x='State code',y='Mortality Rate')
plt.gcf().set_size_inches(10, 5)


# We can see that 'Saxony' has the highest mortality rate of 3.935 among all German states while 'Bremen' has the lowest rate of 2.0. This indicates that the state of Saxony needs to do better covid tracing and treatment to improve the state mortality rate.
#  
# 
# 

# 3. Compare total covid cases in each month
# 
# 
# 
# 
# 
# 

# In[ ]:


cases_per_month=(total_data.set_index('Date'). # use date-time as index
                assign(month=lambda x: x.index.month). # add new column with month
                groupby('month'). # group by that column
                sum() # find sum of cases in a particular month
                ) 
cases_per_month


# In[ ]:


cases_per_month.plot.bar()
plt.yscale("log") # the y scale is logarithmic 
plt.gcf().set_size_inches(10, 5)


# The month of December contributed a large number of covid cases because the lockdown in the country from November was less stringent and there was a delay in case reporting from 'Saxony'(state with large number of covid cases). Note that we took data from beginning of March 2020 to 19th March 2021. Even though we took data for more than one year, December cases remains the highest.

# ### DATA VISUALIZATION OF COVID DATA FOR EACH STATE

# In[ ]:


# function to visualize time series data by taking two state codes as input parameters
def viz(state1,state2):
  df1.plot.line(x='Date',y=[state1,state2])
  plt.gcf().set_size_inches(13, 5)


# 1. North Rhine-Westphalia (DE-NW) and Bavaria (DE-BY)

# In[ ]:


viz('DE-NW','DE-BY')


# 2. Baden-Württemberg (DE-BW) and Saxony (DE-SN)

# In[ ]:


viz('DE-BW','DE-SN')


# 3. Hesse (DE-HE) and Lower Saxony (DE-NI)

# In[ ]:


viz('DE-HE','DE-NI')


# 4. Berlin (DE-BE) and Rhineland-Palatinate (DE-RP)

# In[ ]:


viz('DE-BE','DE-RP')


# 5. Thuringia (DE-TH) and Brandenburg (DE-BB)

# In[ ]:


viz('DE-TH','DE-BB')


# 6. Saxony-Anhalt (DE-ST) and Hamburg (DE-HH)

# In[ ]:


viz('DE-ST','DE-HH')


# 7. Schleswig-Holstein (DE-SH) and Saarland (DE-SL)

# In[ ]:


viz('DE-SH','DE-SL')


# 8. Mecklenburg-Vorpommern (DE-MV) and Bremen (D-HB)

# In[ ]:


viz('DE-MV','DE-HB')


# ### TEST FOR ANOVA 
# 
# 
# 
# 
# 
# 
# 

# We divide the states based on their [Population densities](https://en.wikipedia.org/wiki/List_of_German_states_by_population_density#:~:text=States%20by%20population%20density%20per%20km2%201995-2017%20,%20%2085%20%2013%20more%20rows) into four groups:
# 
# 1.   Group1 : Berlin, Bremen, Hamburg and North Rhine-Westphalia
# 2.   Group2 : Saarland, Baden-Württemberg, Hesse and Saxony
# 3.   Group3 : Rhineland-Palatinate, Bavaria, Schleswig-Holstein and Lower Saxony
# 4.   Group4 : Thuringia, Saxony-Anhalt, Mecklenburg-Vorpommern and Brandenburg
# 
# Division on the basis of population density is because of the fact that coronavirus is more likely to spread in densily populated states.
# 
# **Hypothesis:**
# * Null Hypothesis: There is no difference among groups
# * Alternate Hypothesis: There is difference among atleast one pair of group
# 
# 
# 
# 
# 

# In[ ]:


# converts a 4 column dataframe to linear data
def conv(df):
    s=list()
    cols=df.columns
    for i in range(4):
      s+=list(df[cols[i]])
    return s


# In[ ]:


# random sampling 10 data points from each group
np.random.seed(1234)
dataNew=pd.DataFrame({'Group1':random.sample(conv(df1[['DE-BE','DE-HB','DE-NW','DE-HH']]), 10),
                      'Group2':random.sample(conv(df1[['DE-SL','DE-BW','DE-HE','DE-SN']]), 10),
                      'Group3':random.sample(conv(df1[['DE-RP','DE-BY','DE-SH','DE-NI']]), 10),
                      'Group4':random.sample(conv(df1[['DE-TH','DE-ST','DE-BB','DE-MV']]), 10)})
dataNew


# In[ ]:


# insights into the sample data
dataNew.describe()


# In[ ]:


# Plot number of Corona cases across different groups to check their distribution.
fig = plt.figure(figsize=(10,10))
title = fig.suptitle("Corona cases across different groups", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(2,2,1)
ax1.set_ylabel("Covid Cases") 
sns.kdeplot(dataNew['Group1'], ax=ax1, shade=True,bw=4, color='g')

ax2 = fig.add_subplot(2,2,2)
ax2.set_ylabel("Covid Cases") 
sns.kdeplot(dataNew['Group2'], ax=ax2, shade=True,bw=4, color='y')

ax3 = fig.add_subplot(2,2,3)
ax3.set_ylabel("Covid Cases") 
sns.kdeplot(dataNew['Group3'], ax=ax3, shade=True,bw=4, color='r')

ax4 = fig.add_subplot(2,2,4)
ax4.set_ylabel("Covid Cases") 
sns.kdeplot(dataNew['Group4'], ax=ax4, shade=True,bw=4, color='b')


# In[ ]:


F, p = stats.f_oneway(dataNew['Group1'],dataNew['Group2'],dataNew['Group3'],dataNew['Group4'])
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.3f' % (F, p))


# Since the p-value is greater than the significance level (0.05), we fail to reject the null hypothesis that there is no difference among different groups. 
