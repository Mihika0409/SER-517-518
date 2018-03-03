import pandas as pd

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
print "Number of rows where SBP is lesser than normal: " + str(len(df_age_specific))

df_injury_comments = df[df['Injury Comments'].str.contains("gun", na=False) & (
df['Injury Comments'].str.contains("abdomen", na=False) | df['Injury Comments'].str.contains("chest", na=False) | df[
    'Injury Comments'].str.contains("head", na=False))]
df_injury_comments_level1 = df_injury_comments.loc[df_injury_comments['Levels'].isin(['1'])]
df_injury_comments_level2 = df_injury_comments.loc[df_injury_comments['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_injury_comments_level1))
print "Number of level 2s: " + str(len(df_injury_comments_level2))
print "The size of injury comments data frame is: " + str(len(df_injury_comments))

df_GCS = df[df['Field GCS'] <= 8]
df_GCS_level1 = df_GCS.loc[df_GCS['Levels'].isin(['1'])]
df_GCS_level2 = df_GCS.loc[df_GCS['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_GCS_level1))
print "Number of level 2s: " + str(len(df_GCS_level2))
print "The size of GCS data frame is: " + str(len(df_GCS))

df['ED GCS'] = df['ED GCS'].replace(['*NA', '*ND', '*BL'], value=['15', '15', '15'])
df[['Field GCS', 'ED GCS']] = df[['Field GCS', 'ED GCS']].apply(pd.to_numeric)

df_diff_GCS = df[(df['Field GCS'] - df['ED GCS']) >= 2]
df_diff_GCS_level1 = df_diff_GCS.loc[df_diff_GCS['Levels'].isin(['1'])]
df_diff_GCS_level2 = df_diff_GCS.loc[df_diff_GCS['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(df_diff_GCS_level1))
print "Number of level 2s: " + str(len(df_diff_GCS_level2))
print "Number of rows where difference between the GCS's is greater than 2: " + str(len(df_diff_GCS))

frames = [df_injury_comments, df_GCS, df_diff_GCS, df_age_specific]
result = pd.concat(frames)

print len(result)
result_level1 = result.loc[result['Levels'].isin(['1'])]
result_level2 = result.loc[result['Levels'].isin(['2'])]
print "Number of level 1s: " + str(len(result_level1))
print "Number of level 2s: " + str(len(result_level2))
