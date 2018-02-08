import pandas as pd

data = pd.read_csv("518.csv")

data = data[pd.notnull(data['T1#'])]
data['Received blood within 4 hrs'].fillna('N', inplace=True)
data['Severe Brain Injury'].fillna('N', inplace=True)
data['Field Shock Index'].fillna(0, inplace=True)
data['Field SBP'].fillna(0, inplace=True)
data['Field HR'].fillna(0, inplace=True)
data['Field RR'].fillna(0, inplace=True)
data['Field GCS'].fillna(0, inplace=True)
data['RTS'].fillna(0, inplace=True)
data.to_csv("result.csv")
