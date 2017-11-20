import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample to run stop words.
'''data = "All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(wordsFiltered)'''

# creates a new dataframe with just comments
dataframe = pd.read_csv('Trauma Data.csv')

# dataframe with injury comments and levels
dataframe1 = dataframe[['levels', 'trauma_type', 'feild_GCS', 'age','injury_comments']].dropna(axis=0, how='any')
dataframe1 = dataframe1.rename(columns={'injury_comments': 'Injury Comments'})
dataframe1 = dataframe1.rename(columns={'levels': 'Levels'})
dataframe1 = dataframe1.rename(columns={'feild_GCS': 'GCS'})
dataframe1.to_csv('Comments1.csv')

dataframe2 = dataframe[['levels', 'injury_comments']].dropna(axis=0, how='any')
dataframe2 = dataframe2.rename(columns={'injury_comments': 'Injury Comments'})
dataframe2 = dataframe2.rename(columns={'levels': 'Levels'})

# Finding number of occurrences for each trauma level
dataframe3 = dataframe1.groupby(['Levels']).size()

# Remove stop words. (No, not, etc also being removed atm. Need to fix that)
stopWords = set(stopwords.words('english'))
dataframe1['Injury Comments'] = dataframe1['Injury Comments'].str.lower()# .str.split()
dataframe1['Injury Comments'] = dataframe1['Injury Comments'].apply(lambda x: [item for item in x if item not in stopWords])
dataframe1.to_csv('Comments.csv')

words = ["shock", "spinal", "penetrating injur", "red trauma", "pulseless", "sunset", "wax" ,"arrest", "respiratory distress", "history", "self-inflict", "self inflict", "kill", "attack", "stab", "gun", "intubat", "trigger", "weapon", "atv", "bullet", "gsw", "shot", "unconscious", "hemtoma", "hematomas", "assault", "unrestrain", "head", "airway compromise", "facial injury", "trachea", "amputat", "crushed", "de-glove", "proximal long bon", "skull fracture", "cardiopulmonary"]
for word in words:
    for index, row in dataframe1.iterrows():
        if word in row['Injury Comments']:
            print word, index, row['Levels']

df1 = dataframe1[dataframe1['Levels'].isin(['1'])]
print df1.groupby(['trauma_type']).count()
print df1.groupby(['GCS']).count()

df2 = dataframe1[dataframe1['Levels'].isin(['2'])]
print df2.groupby(['trauma_type']).count()
print df2.groupby(['GCS']).count()

df_isin_1 = dataframe2[dataframe2['Levels'].isin(['1'])]
df_isin_1_2 = df_isin_1['Injury Comments'].str.split(' ', expand=True).stack().value_counts()
df_isin_1_2.to_csv('Level1.csv')

df_isin_2 = dataframe2[dataframe2['Levels'].isin(['2'])]
df_isin_1_2 = df_isin_2['Injury Comments'].str.split(' ', expand=True).stack().value_counts()
df_isin_1_2.to_csv('Level2.csv')


