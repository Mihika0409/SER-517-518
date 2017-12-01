
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

# In[15]:

## Initial Data
# Read the data from the csv
df = pd.read_csv("../Data/Trauma Data.csv")
# Drop rows that have null ti#s (Note: These are rows that have numerous ais 2005#s)
print(df.shape[0])
df = df.dropna(subset=['tid'])

# Rename columns
df = df.rename(columns={'injury_comments': 'Injury Comments'})
df = df.rename(columns={'levels': 'Levels'})
df = df.rename(columns={'feild_SBP': 'Field SBP'})
df = df.rename(columns={'feild_RR': 'Field RR', 'feild_HR': 'Field HR', 'feild_schok_ind': 'Shock Index', 'feild_GCS': 'GCS'})
print(list(df))


# In[3]:

print(df.head())


# In[4]:

# Visualize the sbp and rr columns
def coerce_df_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')

# Convert t1, and sbp/rr to numeric. Int if possible.
# Note: RR and SBP cannot be converted to int atm since they may have na's
coerce_df_columns_to_numeric(df, ['tid', 'Field SBP', 'Field RR'])

# Get a subset of the dataframe to visualize our desired columns
df_sbprr_subset = pd.DataFrame(df, columns = ['tid', 'Field SBP', 'Field RR'])
print(df_sbprr_subset.head())


# In[5]:

# See unique SBP fields. Quick visualization for nulls.
# Safe to assume Assuming that sbf fields are positive integers in a small range.
init_sbp_arr = np.sort(df['Field SBP'].unique())
init_sbp_arr


# In[6]:

# See unique RR fields. Same reasoning as SBP fields
init_rr_arr = np.sort(df['Field RR'].unique())
print(init_rr_arr)


# In[7]:

# Respiratory rate subset
rr_subset = df
rr_subset = rr_subset.dropna(subset=['Field RR'])
rr_subset = rr_subset.ix[(rr_subset['Field RR']> 0) & (rr_subset['Field RR'] != 9999)]
#Test that sbp doesn't have nan
final_rr_arr = np.sort(rr_subset['Field RR'].unique().astype(int))
print(final_rr_arr)


# In[8]:

# SBP subset
sbp_subset = df
sbp_subset = sbp_subset.dropna(subset=['Field SBP'])
sbp_subset = sbp_subset.ix[(sbp_subset['Field SBP']> 0) & (rr_subset['Field SBP'] != 9999)]
#Test that sbp doesn't have nan
final_sbp_arr = np.sort(sbp_subset['Field SBP'].unique().astype(int))
print(final_sbp_arr)


# In[9]:

# See how many rows were dropped when removing nan rr's and nan sbp's
df_count = df.shape[0]
rr_count = rr_subset.shape[0]
sbp_count = sbp_subset.shape[0]
missing_rr_count = df_count - rr_count
missing_sbp_count = df_count - sbp_count
print('Missing Data Analysis:')
print('Clean rows- Rows without 0, nan, 9999, or other null values for SBP/RR')
print('# of df rows:', df_count)
print('# of valid rr rows:', rr_count)
print('# of valid sbp rows:', sbp_count)
print('# of invalid rr rows:', missing_rr_count)
print('# of invalid sbp rows:', missing_sbp_count)
print('percentage of invalid rr rows: {0:.3f}%'.format(100 * missing_rr_count / df_count))
print('percentage of invalid sbp rows: {0:.3f}%'.format(100 * missing_sbp_count / df_count))

# High, low, healthy groups.. SBP: < 120, 120-140, >140   RR: < 12, 12-20, >20... For each group, determine % of lvl 1 and 2
# Different groups.. level of patients in normal group, low sbp: high/low rr, high sbp: high/low rr (include normals?)

def rows_in_range(data_frame, column_name, low_limit, high_limit):
    low_range = data_frame.ix[(data_frame[column_name] < low_limit)]
    norm_range = data_frame.ix[(data_frame[column_name] >= low_limit) & (data_frame[column_name] <= high_limit)]
    high_range = data_frame.ix[(data_frame[column_name] > high_limit)]
    return [low_range, norm_range, high_range]

def print_subset_analysis(subset_range_arr, column_alias):
    low_range = subset_range_arr[0]
    norm_range = subset_range_arr[1]
    high_range = subset_range_arr[2]
    # Refactoring needed if this has to be expanded. for loop object or 2d array.(low priority since this is for a one time alaysis report)
    low_range_one = low_range[low_range['Levels'].isin(['1'])].shape[0]
    low_range_two = low_range[low_range['Levels'].isin(['2'])].shape[0]
    norm_range_one = norm_range[norm_range['Levels'].isin(['1'])].shape[0]
    norm_range_two = norm_range[norm_range['Levels'].isin(['2'])].shape[0]
    high_range_one = high_range[high_range['Levels'].isin(['1'])].shape[0]
    high_range_two = high_range[high_range['Levels'].isin(['2'])].shape[0]
    # Total # of patients in any category. Note: They have to be either high, normal, or medium.
    total_one = low_range_one + norm_range_one + high_range_one
    total_two = low_range_two + norm_range_two + high_range_two
    print('\n\n', column_alias, 'analysis')
    print('# of', column_alias, 'rows in the following categories: low-', low_range.shape[0], 'norm-', norm_range.shape[0], 'high-', high_range.shape[0])
    print('low', column_alias ,'patients that have level 1 trauma: ', low_range_one, '({0:.3f}%)'.format(100 * low_range_one / low_range.shape[0]))
    print('low', column_alias ,'patients that have level 2 trauma: ', low_range_two, '({0:.3f}%)'.format(100 * low_range_two / low_range.shape[0]))
    print('normal', column_alias ,'patients that have level 1 trauma: ', norm_range_one, '({0:.3f}%)'.format(100 * norm_range_one / norm_range.shape[0]))
    print('normal', column_alias ,'patients that have level 2 trauma: ', norm_range_two, '({0:.3f}%)'.format(100 * norm_range_two / norm_range.shape[0]))
    print('high', column_alias ,'patients that have level 1 trauma: ', high_range_one, '({0:.3f}%)'.format(100 * high_range_one / high_range.shape[0]))
    print('high', column_alias ,'patients that have level 2 trauma: ', high_range_two, '({0:.3f}%)'.format(100 * high_range_two / high_range.shape[0]))
    
    print('% of level 1 patients that have a', column_alias, 
          ' in a range that is: low: {0:.3f}%,'.format(100 * low_range_one / total_one), 
          'normal- {0:.3f}%,'.format(100 * norm_range_one / total_one), 
          'high- {0:.3f}%'.format(100 * high_range_one / total_one))
    print('% of level 2 patients that have a', column_alias, 
      ' in a range that is: low: {0:.3f}%,'.format(100 * low_range_two / total_two), 
      'normal- {0:.3f}%,'.format(100 * norm_range_two / total_two), 
      'high- {0:.3f}%'.format(100 * high_range_two / total_two))

    
sbp_range = rows_in_range(sbp_subset, 'Field SBP', 120, 140)
rr_range = rows_in_range(rr_subset, 'Field RR', 12, 20)
print_subset_analysis(sbp_range, 'SBP')
print_subset_analysis(rr_range, 'RR')


# In[13]:

# RR subset, showing sbp and rr (note, sbp can be null/0)
print(pd.DataFrame(rr_subset, columns = ['T1#', 'Field SBP', 'Field RR']))


# In[14]:

# RR subset, showing sbp and rr (note, rr can be null/0)
print(pd.DataFrame(sbp_subset, columns = ['T1#', 'Field SBP', 'Field RR']))


# In[12]:

# Download Respiratory Rate and Systolic Blood Pressure subsets as csv files
rr_subset.to_csv('../Data/rr_subset.csv')
sbp_subset.to_csv('../Data/sbp_subset.csv', encoding='utf-8', index=False)


# In[ ]:



