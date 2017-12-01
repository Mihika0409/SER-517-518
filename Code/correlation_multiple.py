import matplotlib.pyplot as plt
import unicodecsv

import numpy as np
import pandas as pd
import seaborn as sb
from xlrd import open_workbook


# Converts the xls file to csv
def xls2csv (xls_filename, csv_filename):

    wb = open_workbook(xls_filename)
    sh = wb.sheet_by_index(0)
    fh = open(csv_filename,"wb")
    csv_out = unicodecsv.writer(fh, encoding='utf-8')
    for row_number in xrange (sh.nrows):
        csv_out.writerow(sh.row_values(row_number))
    fh.close()
    return csv_filename
csv_file = xls2csv('Copy of Trauma Data Sample From Jan 2016 to Jan 2017.xlsx','csv.csv')

df = pd.read_csv('/Users/vc/Downloads/refined7 (1).csv')
df['Airbag Deployment'] = df['Airbag Deployment'].replace('*BL',0)
df['Airbag Deployment'] = df['Airbag Deployment'].replace('*NA',0)
df['Airbag Deployment'] = df['Airbag Deployment'].replace('*ND',0)
for index, row in df.iterrows():
    data = df['GCS']
    #print data
# creates a new dataframe with just comments
'''dfn = pd.read_csv('csv.csv')
df1 = dfn.ix[:, 19:20].dropna(axis=0, how='any')
df1 = df1.rename(columns={'Unnamed: 19': 'Comments'})

for index, row in df1.iterrows():
    dataC = df1['Comments']
    print dataC'''
df['Gender'] = (df['Gender'] !='M').astype(int)
df
for index, row in df.iterrows():
    data = df['Gender']
#    print data
cols = ['Gender','Transport Mode','GCS','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
sb.set(font_scale=1.5)
cor_heat_map = sb.heatmap(cor_matrix, cbar=True, annot=True,square=True,fmt='.2f', annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()
