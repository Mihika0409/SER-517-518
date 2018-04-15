import pandas as pd

fields = ['Age in Years', 'Gender', 'MV Speed','RTS','Field GCS','Field SBP', 'Field HR', 'Field RR','Intubated','Resp Assistance']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/m4.csv',
    usecols=fields)
fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/m4.csv',
    usecols=fields1)
df['RTS'] = np.where(((df['RTS'] == '*NA') | (df['RTS'] == '*ND') | (df['RTS'] == '*BL') | (df['RTS'] == '')) & (df['Gender'] == 1),'6.15',df['RTS'])
df['RTS'] = np.where(((df['RTS'] == '*NA') | (df['RTS'] == '*ND') | (df['RTS'] == '*BL') | (df['RTS'] == '')) & (df['Gender'] == 0),'7.75',df['RTS'])
df['Field GCS'] = np.where(((df['Field GCS'] == '*NA') | (df['Field GCS'] == '*ND') | (df['Field GCS'] == '*BL') | (df['Field GCS'] == '')) & (df['Gender'] == 1),'10.24',df['Field GCS'])
df['Field GCS'] = np.where(((df['Field GCS'] == '*NA') | (df['Field GCS'] == '*ND') | (df['Field GCS'] == '*BL') | (df['Field GCS'] == '')) & (df['Gender'] == 0),'14.90',df['Field GCS'])
df['Field SBP'] = np.where(((df['Field SBP'] == '*NA') | (df['Field SBP'] == '*ND') | (df['Field SBP'] == '*BL') | (df['Field SBP'] == '')) & (df['Gender'] == 1),'111.50',df['Field SBP'])
df['Field SBP'] = np.where(((df['Field SBP'] == '*NA') | (df['Field SBP'] == '*ND') | (df['Field SBP'] == '*BL') | (df['Field SBP'] == '')) & (df['Gender'] == 0),'120.25',df['Field SBP'])
df['Field HR'] = np.where(((df['Field HR'] == '*NA') | (df['Field HR'] == '*ND') | (df['Field HR'] == '*BL') | (df['Field HR'] == '')) & (df['Gender'] == 1),'106',df['Field HR'])
df['Field HR'] = np.where(((df['Field HR'] == '*NA') | (df['Field HR'] == '*ND') | (df['Field HR'] == '*BL') | (df['Field HR'] == '')) & (df['Gender'] == 0),'110',df['Field HR'])
df['MV Speed'] = df['MV Speed'].replace(['*NA','*ND','*BL',''],'0')

print df
