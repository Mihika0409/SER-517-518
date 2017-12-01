import pandas as pd
import numpy as np

import scipy.stats as sp
import csv
import matplotlib.pyplot as plt

#To find the pearson and spearmanr corelation between The motor vehicle speed and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv')
cols = ['MV Speed','Levels']
cols1 = ['MV Speed']
cols2 = ['Levels']
cor_matrix1 = np.corrcoef(df[cols].values.T)
cor_matrix = sp.spearmanr(df[cols1].values,df[cols2].values)
print "Mv Speed and Trauma Level"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The SBP and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/sbp.csv')
cols = ['SBP','Levels']
cols1 = ['SBP']
cols2 = ['Levels']
cor_matrix = sp.spearmanr(df[cols1].values,df[cols2].values)
cor_matrix1 = np.corrcoef(df[cols].values.T)
print "SBP and Trauma Level"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The GCS and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/gcs.csv')
cols = ['GCS','Levels']
cols1 = ['GCS']
cols2 = ['Levels']
cor_matrix1 = sp.spearmanr(df[cols1].values,df[cols2].values)
cor_matrix = np.corrcoef(df[cols].values.T)
print "GCS and Trauma Level"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The Respiratory Rate and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/rr.csv')
cols = ['RR','Levels']
cols1 = ['RR']
cols2 = ['Levels']
cor_matrix1 = sp.spearmanr(df[cols1].values,df[cols2].values)
cor_matrix = np.corrcoef(df[cols].values.T)
print "RR and Trauma Level"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The Respiratory Rate and the trauma level.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/Pulse.csv')
cols = ['Pulse','Levels']
cols1 = ['Pulse']
cols2 = ['Levels']
cor_matrix1 = sp.spearmanr(df[cols1].values,df[cols2].values)
cor_matrix = np.corrcoef(df[cols].values.T)
print "Pulse and Trauma Level"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The motor vehicle speed and the GCS.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/GCSmv.csv')
cols = ['MV Speed','GCS']
cols1 = ['MV Speed']
cols2 = ['GCS']
cor_matrix1 = np.corrcoef(df[cols].values.T)
cor_matrix = sp.spearmanr(df[cols1].values,df[cols2].values)
print "Mv Speed and GCS"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The motor vehicle speed and the RR.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/GCSrr.csv')
cols = ['MV Speed','RR']
cols1 = ['MV Speed']
cols2 = ['RR']
cor_matrix1 = np.corrcoef(df[cols].values.T)
cor_matrix = sp.spearmanr(df[cols1].values,df[cols2].values)
print "Mv Speed and RR"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The motor vehicle speed and the Pulse.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/GCSpulse.csv')
cols = ['MV Speed','Pulse']
cols1 = ['MV Speed']
cols2 = ['Pulse']
cor_matrix1 = np.corrcoef(df[cols].values.T)
cor_matrix = sp.spearmanr(df[cols1].values,df[cols2].values)
print "Mv Speed and Pulse"
print cor_matrix
print cor_matrix1

#To find the pearson and spearmanr corelation between The motor vehicle speed and the SBP.
df = pd.read_csv('/Users/satishnandan/Desktop/TraumaActivation/GCSsbp.csv')
cols = ['MV Speed','SBP']
cols1 = ['MV Speed']
cols2 = ['SBP']
cor_matrix1 = np.corrcoef(df[cols].values.T)
cor_matrix = sp.spearmanr(df[cols1].values,df[cols2].values)
print "Mv Speed and SBP"
print cor_matrix
print cor_matrix1


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

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/GCSmv.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[25].strip() == "*NA" or row[25].strip() == "*ND" or row[25].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/GCSrr.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[24].strip() == "*NA" or row[24].strip() == "*ND" or row[24].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/GCSPulse.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[23].strip() == "*NA" or row[23].strip() == "*ND" or row[23].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#will remove not available, no data and blank entries from data set.
with open('/Users/satishnandan/Desktop/TraumaActivation/mv2.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/GCSsbp.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[22].strip() == "*NA" or row[22].strip() == "*ND" or row[22].strip() == "*BL":
            continue
        else:
            writer.writerow(row)
