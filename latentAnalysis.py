import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis

fields = ['Age in Years', 'Gender', 'MV Speed','Resp Assistance','RTS','Field GCS','Field SBP', 'Field HR', 'Field RR']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeed10.csv',
    usecols=fields)
fields1 = ['Total LOS (ED+Admit)']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeed10.csv',
    usecols=fields1)
factor = FactorAnalysis(n_components=1, random_state=101).fit(df.transpose())
frame = pd.DataFrame(factor.components_).transpose()
print Y.join(frame).shape
cor_matrix = np.corrcoef(Y.T.values, frame.T.values)
print cor_matrix
