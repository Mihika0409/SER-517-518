
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from xlrd import open_workbook
import unicodecsv
import csv
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv')
cols = ['MV Speed','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
print cor_matrix

with open('/Users/satishnandan/Desktop/TraumaActivation/test.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/airbag.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[15].strip() == "*NA" or row[15].strip() == "*ND" or row[15].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

count = 0
count1 = 0
count2 = 0
with open('/Users/satishnandan/Desktop/TraumaActivation/new.csv', 'rU') as inp:
    for row in csv.reader(inp):
        if row[9].strip() == "*NA" or row[9].strip() == "*ND" or row[9].strip() == "*BL":
            continue
        else:
            if row[4].strip() == "1":
                count = count + 1
            elif row[4].strip() == "2":
                count1 = count1 + 1

print "Trauma Level 1 are: " + str(count)
print "Trauma Level 2 are:" + str(count1)
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# Sample to run stop words.
'''data = "All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(wordsFiltered)'''


# Converts the xls file to csv
def xls2csv(xls_filename, csv_filename):
    wb = open_workbook(xls_filename)
    sh = wb.sheet_by_index(0)
    fh = open(csv_filename, "wb")
    csv_out = unicodecsv.writer(fh, encoding='utf-8')
    for row_number in xrange(sh.nrows):
        csv_out.writerow(sh.row_values(row_number))
    fh.close()
    return csv_filename


csv_file = xls2csv('/Users/satishnandan/Desktop/TraumaActivation/Copy of Stacked Trauma data.xlsx', '/Users/satishnandan/Desktop/TraumaActivation/newData.csv')

# creates a new dataframe with just comments
df = pd.read_csv('csv.csv')
df1 = df.ix[:, 19:20].dropna(axis=0, how='any')
df1 = df1.rename(columns={'Unnamed: 19': 'Comments'})

for index, row in df1.iterrows():
    print row['Comments']
headers = ['Levels','MV Speed']
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv')
x = df['MV Speed']
y = df['Levels']
x = x.astype(float)
y = y.astype(float)
plt.plot(y,x,".")
plt.ylim(50,200)
plt.xlim(0,3)
plt.show(block=True)
dataset = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/newData.csv', header=None)
temp = range(0,42)
# mark zero values as missing or NaN
dataset[temp] = dataset[temp].replace("*BL", numpy.NaN)
# drop rows with missing values
dataset.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(dataset.shape)
count1 = 0
with open('/Users/satishnandan/Desktop/TraumaActivation/new.csv', 'rU') as inp:
    for row in csv.reader(inp):
        if row[13].strip() == "*NA" or row[13].strip() == "*ND" or row[13].strip() == "*BL" or row[13].strip() == "":
            continue
        else:
            if row[14].strip() == "*NA" or row[14].strip() == "*ND" or row[14].strip() == "*BL" or row[14].strip() == "":
                continue
            else:
                print row

                count1 = count1 + 1

with open('/Users/satishnandan/Desktop/TraumaActivation/new.csv', 'rU') as inp:
    for row in csv.reader(inp):
        if row[13].strip() == "*NA" or row[13].strip() == "*ND" or row[13].strip() == "*BL" or row[13].strip() == "":
            continue
        else:
            if row[38].strip() == "D":
                dead = dead + 1
            elif row[38].strip() == "L":
                alive = alive + 1
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

with open('/Users/satishnandan/Desktop/TraumaActivation/new.csv', 'rU') as inp:
    for row in csv.reader(inp):
        if row[13].strip() == "*NA" or row[13].strip() == "*ND" or row[13].strip() == "*BL" or row[13].strip() == "":
            continue
        else:
            if is_number(row[19]):
                if float(row[19]) < 0.75:
                    count = count + 1
                    if row[4] == "1":
                        count1 = count1 + 1
burn = 0
penetrating = 0
blunt = 0
other = 0

with open('/Users/satishnandan/Desktop/TraumaActivation/new.csv', 'rU') as inp:
    for row in csv.reader(inp):
        if row[13].strip() == "*NA" or row[13].strip() == "*ND" or row[13].strip() == "*BL" or row[13].strip() == "":
            continue
        else:
            if row[6].strip() == "Burn":
                burn = burn + 1
            elif row[6].strip() == "Penetrating":
                penetrating = penetrating + 1
            elif row[6].strip() == "Blunt":
                blunt = blunt + 1
            else:
                print row[6]
                other = other + 1
print burn
print penetrating
print blunt
print other
