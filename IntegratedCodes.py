import pandas as pd

df = pd.read_csv("ICD.csv")

'''df_injury = df['Injury Comments']
df_injury = df_injury[pd.notnull(df['Injury Comments'])]

df_injury.to_csv("Injury.csv")'''

print df
