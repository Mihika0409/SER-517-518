import pandas as pd
from sklearn.decomposition import FactorAnalysis

df = pd.read_csv("result.csv")

df = df[['Age in Years', 'Gender', 'Trauma Type', 'MV Speed', 'Fall Height', 'Transport Mode', 'Field SBP', 'Field HR',
         'Field Shock Index', 'Field RR', 'Resp Assistance',
         'RTS', 'Field GCS', 'Arrived From', 'Received blood within 4 hrs', 'Time to 1st OR Visit (mins.)',
         'ED LOS (mins)', 'Total LOS (ED+Admit)']]

columns = df[
    ['Age in Years', 'Gender', 'Trauma Type', 'MV Speed', 'Fall Height', 'Transport Mode', 'Field SBP', 'Field HR',
     'Field Shock Index', 'Field RR', 'Resp Assistance',
     'RTS', 'Field GCS', 'Arrived From', 'Received blood within 4 hrs', 'Time to 1st OR Visit (mins.)', 'ED LOS (mins)',
     'Total LOS (ED+Admit)']]

for x in columns:
    df = df[pd.notnull(df[x])]

dummy = pd.get_dummies(df['Arrived From'])
df = pd.concat([df, dummy], axis=1)
del df['Arrived From']

dummy1 = pd.get_dummies(df['Transport Mode'])
df = pd.concat([df, dummy1], axis=1)
del df['Transport Mode']

dummy3 = pd.get_dummies(df['Trauma Type'])
df = pd.concat([df, dummy3], axis=1)
del df['Trauma Type']

dummy4 = pd.get_dummies(df['Gender'])
df = pd.concat([df, dummy4], axis=1)
del df['Gender']

df[['Fall Height', 'MV Speed']] = df[['Fall Height', 'MV Speed']].replace(['*NA', '*ND', '*BL'], [0, 0, 0])
df[['Received blood within 4 hrs']] = df[['Received blood within 4 hrs']].replace(['N'], [0])
df[['Received blood within 4 hrs']] = df[['Received blood within 4 hrs']].replace(['Y'], [1])
df[['Field Shock Index']] = df[['Field Shock Index']].replace(['#VALUE!'], [0])
df[['Time to 1st OR Visit (mins.)']] = df[['Time to 1st OR Visit (mins.)']].replace(['*BL', '*NA'], [99999, 99999])
df[['ED LOS (mins)']] = df[['ED LOS (mins)']].replace(['*BL', '*NA'], [0, 0])

X = df[['Age in Years', 'MV Speed', 'Fall Height', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR',
        'Resp Assistance', 'RTS', 'Field GCS',
        'Received blood within 4 hrs', 'Time to 1st OR Visit (mins.)', 'ED LOS (mins)', 'Total LOS (ED+Admit)', '*BL',
        '*ND',
        'Clinic/MD Office', 'Home', 'Other', 'Referring Hospital', 'Scene of Injury', 'Urgent Care',
        'Fixed-wing Ambulance', 'Ground Ambulance',
        'Helicopter Ambulance', 'Other', 'Private/Public Vehicle/Walk-in', '*NA', '*ND', 'Blunt', 'Burn', 'Penetrating',
        'F', 'M']]

list = []
list = X.values.tolist()

factor = FactorAnalysis().fit(list)
print factor

print factor.get_covariance()
print factor.get_params()
print factor.get_precision()
