import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("result2.csv")
df.columns = ['T1', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD', 'Trauma Type',
              'Report of physical abuse', 'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle',
              'Safet Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type',
              'Transport Mode', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'Resp Assistance',
              'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'Dispositon from  ED', 'ED SBP', 'ED HR', 'ED RR',
              'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
              'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive', 'Discharge Disposition', 'Injury Severity Score', 'AIS 2005']

df_age_specific = df[(((df['Age in Years'] >= 4) & (df['Age in Years'] <= 6)) & (df['Field SBP'] < 90)) | (
((df['Age in Years'] >= 7) & (df['Age in Years'] <= 16)) & (df['Field SBP'] < 100))]
df_age_specific_level1 = df_age_specific.loc[df_age_specific['Levels'].isin(['1'])]
df_age_specific_level2 = df_age_specific.loc[df_age_specific['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_age_specific_level1))
print "Number of level 2s: " + str(len(df_age_specific_level2))
print "Number of rows where SBP is lesser than normal: " + str(len(df_age_specific)) + "\n"

df_respiratory_obstruction = df[df['Injury Comments'].str.contains("intubate", na=False)]

df_respiratory_obstruction_transfer = df_respiratory_obstruction[
    df_respiratory_obstruction['Arrived From'] == 'Referring Hospital']
df_respiratory_obstruction_transfer_level1 = df_respiratory_obstruction_transfer.loc[
    df_respiratory_obstruction_transfer['Levels'].isin(['1'])]
df_respiratory_obstruction_transfer_level2 = df_respiratory_obstruction_transfer.loc[
    df_respiratory_obstruction_transfer['Levels'].isin(['2'])]

df_respiratory_obstruction_soi = df_respiratory_obstruction[
    df_respiratory_obstruction['Arrived From'] == 'Scene of Injury']
df_respiratory_obstruction_soi_level1 = df_respiratory_obstruction_soi.loc[
    df_respiratory_obstruction_soi['Levels'].isin(['1'])]
df_respiratory_obstruction_soi_level2 = df_respiratory_obstruction_soi.loc[
    df_respiratory_obstruction_soi['Levels'].isin(['2'])]

df_respiratory_obstruction_level1 = df_respiratory_obstruction.loc[df_respiratory_obstruction['Levels'].isin(['1'])]
df_respiratory_obstruction_level2 = df_respiratory_obstruction.loc[df_respiratory_obstruction['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_respiratory_obstruction_level1))
print "Number of level 2s: " + str(len(df_respiratory_obstruction_level2))
print "Number of level 1s with arrival from referring hospital: " + str(len(df_respiratory_obstruction_transfer_level1))
print "Number of level 2s with arrival from referring hospital: " + str(len(df_respiratory_obstruction_transfer_level2))
print "Number of level 1s with arrival from scene of injury: " + str(len(df_respiratory_obstruction_soi_level1))
print "Number of level 2s with arrival from scene of injury: " + str(len(df_respiratory_obstruction_soi_level2))
print "The size of respiratory obstruction data frame is: " + str(len(df_respiratory_obstruction)) + "\n"

df_injury_comments = df[df['Injury Comments'].str.contains("gun", na=False) & (
df['Injury Comments'].str.contains("abdomen", na=False) | df['Injury Comments'].str.contains("chest", na=False) | df[
    'Injury Comments'].str.contains("head", na=False))]
df_injury_comments_level1 = df_injury_comments.loc[df_injury_comments['Levels'].isin(['1'])]
df_injury_comments_level2 = df_injury_comments.loc[df_injury_comments['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_injury_comments_level1))
print "Number of level 2s: " + str(len(df_injury_comments_level2))
print "The size of gun shots data frame is: " + str(len(df_injury_comments)) + "\n"

df_GCS = df[df['Field GCS'] <= 8]
df_GCS_level1 = df_GCS.loc[df_GCS['Levels'].isin(['1'])]
df_GCS_level2 = df_GCS.loc[df_GCS['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_GCS_level1))
print "Number of level 2s: " + str(len(df_GCS_level2))
print "The size of GCS data frame is: " + str(len(df_GCS)) + "\n"

df['ED GCS'] = df['ED GCS'].replace(['*NA', '*ND', '*BL'], value=['15', '15', '15'])
df[['Field GCS', 'ED GCS']] = df[['Field GCS', 'ED GCS']].apply(pd.to_numeric)

df_diff_GCS = df[(df['Field GCS'] - df['ED GCS']) >= 2]
df_diff_GCS_level1 = df_diff_GCS.loc[df_diff_GCS['Levels'].isin(['1'])]
df_diff_GCS_level2 = df_diff_GCS.loc[df_diff_GCS['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_diff_GCS_level1))
print "Number of level 2s: " + str(len(df_diff_GCS_level2))
print "Number of rows where difference between the GCS's is greater than 2: " + str(len(df_diff_GCS)) + "\n"

frames = [df_injury_comments, df_GCS, df_diff_GCS, df_age_specific]
result = pd.concat(frames)
result_level1 = result.loc[result['Levels'].isin(['1'])]
result_level2 = result.loc[result['Levels'].isin(['2'])]
print "Total results in ACS6 criteria: " + str(len(result))
print "Number of level 1s: " + str(len(result_level1))
print "Number of level 2s: " + str(len(result_level2)) + "\n"

print "Total patients: " + str(len(df))
print "Total number of level 1 patients: " + str(len(df.loc[df['Levels'].isin(['1'])]))
print "Total number of level 2 patients: " + str(len(df.loc[df['Levels'].isin(['2'])]))

print "Accuracy of ACS6 is: " + str(len(result_level1) * 1.0 / len(df.loc[df['Levels'].isin(['1'])]))

print "------------------------------------------------------------------------------------------"

df = pd.read_csv("result.csv")
del df['Unnamed: 0']

df = df[
    ['Levels', 'T1#', 'Age in Years', 'Gender', 'Trauma Type', 'Report of physical abuse?', 'MV Speed', 'Fall Height',
     'Transport Mode', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS',
     'Arrived From']]

columns = ['Levels', 'T1#', 'Age in Years', 'Gender', 'Trauma Type', 'Report of physical abuse?', 'MV Speed',
           'Fall Height', 'Transport Mode', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'Resp Assistance',
           'RTS', 'Field GCS', 'Arrived From']
for x in columns:
    df = df[pd.notnull(df[x])]
df = df.loc[df['Levels'].isin(['1', '2'])]
df[['Levels']] = df[['Levels']].replace([2], [0])

dummy = pd.get_dummies(df['Arrived From'])
df = pd.concat([df, dummy], axis=1)
del df['Arrived From']

dummy1 = pd.get_dummies(df['Transport Mode'])
df = pd.concat([df, dummy1], axis=1)
del df['Transport Mode']

dummy2 = pd.get_dummies(df['Report of physical abuse?'])
df = pd.concat([df, dummy2], axis=1)
del df['Report of physical abuse?']

dummy3 = pd.get_dummies(df['Trauma Type'])
df = pd.concat([df, dummy3], axis=1)
del df['Trauma Type']

dummy4 = pd.get_dummies(df['Gender'])
df = pd.concat([df, dummy4], axis=1)
del df['Gender']

df[['Fall Height', 'MV Speed']] = df[['Fall Height', 'MV Speed']].replace(['*NA', '*ND', '*BL'], [0, 0, 0])
df[['Levels']] = df[['Levels']].replace(['N'], [4])
df[['Field Shock Index']] = df[['Field Shock Index']].replace(['#VALUE!'], [0])

X = df.values[:, 1:len(df)]
Y = df.values[:, 0:1]
Y = Y.astype('int')

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "presort": [True, False],
              "criterion": ["gini", "entropy"]}

y = Y.reshape(-1, )

clf = DecisionTreeClassifier()
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, return_train_score=True,
                                   refit=True, cv=10)
random_search.fit(X, y)

print "Random search score: "
print random_search.score(X, y)

print "Accuracy of the model: "
print random_search.best_score_
