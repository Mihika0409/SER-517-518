import pandas as pd

# df = pd.read_csv("ICD.csv")

'''df_injury = df['Injury Comments']
df_injury = df_injury[pd.notnull(df['Injury Comments'])]

df_injury.to_csv("Injury.csv")'''

'''df_fall = df[df['ICD'].str.contains("FALL", na=False)]
df_fall.to_csv("fall.csv")

df_notfall = df[~df['ICD'].str.contains("FALL", na=False)]
df_notfall.to_csv("NotFall.csv")

df_mv = df_notfall[df_notfall['ICD'].str.contains("MOTOR", na=False) | df_notfall['ICD'].str.contains("CYCLE", na=False) | df_notfall['ICD'].str.contains("CAR", na=False)]
df_mv.to_csv("MotorVehicle.csv")

df_notMV = df_notfall[~df_notfall['ICD'].str.contains("MOTOR", na=False) & ~df_notfall['ICD'].str.contains("CYCLE", na=False) & ~df_notfall['ICD'].str.contains("CAR", na=False)]
df_notMV.to_csv("NotMV.csv")

print len(df_fall)
print len(df_mv)
print len(df_notMV)'''

df = pd.read_csv("Injury.csv")

df.columns = ['T1', 'Comments']

for index, item in df['Comments'].iteritems():
    if row['Comments'].str.contains("fall", na=False):
        print "Fall" + row['T1']
    if row['Comments'].str.contains("motor vehicle", na=False):
        print "MV" + row['T1']