import pandas as pd
from xlrd import open_workbook
import unicodecsv

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Sample to run stop words.
'''data = "All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)'''

#Converts the xls file to csv
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

#creates a new dataframe with just comments
df = pd.read_csv('csv.csv')
df1 = df.ix[:, 19:20].dropna(axis=0, how='any')
df1 = df1.rename(columns = {'Unnamed: 19':'Comments'})

for index, row in df1.iterrows():
    print row['Comments']