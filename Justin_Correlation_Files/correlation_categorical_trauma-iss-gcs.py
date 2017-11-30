
# coding: utf-8

# In[9]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:

df = pd.read_csv("Copy of Stacked Trauma data.csv")
df2 = pd.read_csv("Original_download_TA_2017_refactored.csv")
df3 = pd.read_csv("refined2.csv")
copy_stack = list(df)
original_stack = list(df2)
print(copy_stack)


# In[11]:

list(df3)


# In[12]:
print("csv comparison:")
print("copy stack len", len(copy_stack), "copy cols that exist in original:")
print(np.in1d(copy_stack,original_stack))
print("original stack len", len(original_stack), "original cols that exist in copy:")
# Check which copy columns exist in the original stack



# In[13]:

# Check which original_stack columns exist in the copy stack
print(np.in1d(original_stack, copy_stack))


# In[14]:

print('Column Types:')
print(df3.dtypes)


# In[15]:

# plt.matshow(df.corr())
# plt.show()
import seaborn as sns

df_new = df3
def splitColumn(dataframe, type, colPrefix):
    if colPrefix != None:
        df_dummies = pd.get_dummies(dataframe[type], prefix=colPrefix)
    else:
        df_dummies = pd.get_dummies(dataframe[type])
    dataframe = pd.concat([dataframe, df_dummies], axis=1)
    del dataframe[type]
    return dataframe
# TODO: Refactor this. Just use an array.
df_new = splitColumn(df_new, 'Trauma Type', None)
df_new = splitColumn(df_new, 'Levels', 'Trauma Category')
df_new.dropna
# df_new = splitColumn(df_new, 'Transport Mode', None
# df_new = removeType(df_new, 'Gender')


# Clean the data (remove )
def removeRows(dataframe, columnArray, removalValues):
    modified_df = dataframe
    for i in range(len(columnArray)):
        for j in range(len(removalValues)):
            modified_df = modified_df[modified_df[columnArray[i]].str.contains(removalValues[j]) == False]
    return modified_df

df_new = removeRows(df_new, ['Injury Severity Score', 'GCS'], ['NA', 'BL', 'ND'])

df_new[['Injury Severity Score','GCS']] = df_new[['Injury Severity Score','GCS']].apply(pd.to_numeric)
print("Table with null ISS/GCS Rows Remvoed:")
print(df_new.head())


# In[16]:

# - Create a correlation matrix to quantify the linear relationships between features.
# In[3]:
import numpy as np
cols = ['Blunt', 'Burn', 'Penetrating', 'Trauma Category_1', 'Trauma Category_2', 'Injury Severity Score', 'GCS']
cor_matrix = np.corrcoef(df_new[cols].values.T) # note we transpose to get the data by columns. Columns become rows.
sns.set(font_scale=.5)
cor_heat_map = sns.heatmap(cor_matrix,
cbar=True,
annot=True,
square=True,
fmt='.2f',
annot_kws={'size':15},
yticklabels=cols,
xticklabels=cols)
sns.plt.show()



