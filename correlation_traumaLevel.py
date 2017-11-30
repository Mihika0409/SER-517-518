import pandas as pd
import numpy as np
import csv

#To find the pearson corelation between The motor vehicle speed and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv')
cols = ['MV Speed','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
print cor_matrix

#To find the pearson corelation between The SBP and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/sbp.csv')
cols = ['SBP','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
print cor_matrix

#To find the pearson corelation between The GCS and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/gcs.csv')
cols = ['GCS','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
print cor_matrix

#To find the pearson corelation between The Respiratory Rate and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/rr.csv')
cols = ['RR','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
print cor_matrix

#To find the pearson corelation between The Pulse  and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/Pulse.csv')
cols = ['Pulse','Levels']
cor_matrix = np.corrcoef(df[cols].values.T)
print cor_matrix

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/test.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/Pulse.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[23].strip() == "*NA" or row[23].strip() == "*ND" or row[23].strip() == "*BL":
            continue
        else:
            writer.writerow(row)


#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/test.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[18].strip() == "*NA" or row[18].strip() == "*ND" or row[18].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/test.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/sbp.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[22].strip() == "*NA" or row[22].strip() == "*ND" or row[22].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/test.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/gcs.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[25].strip() == "*NA" or row[25].strip() == "*ND" or row[25].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/test.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/rr.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[24].strip() == "*NA" or row[24].strip() == "*ND" or row[24].strip() == "*BL":
            continue
        else:
            writer.writerow(row)
