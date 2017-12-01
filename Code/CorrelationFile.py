import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sb

# Case 1 - Airbag Deployment with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Airbag Deployment', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Airbag Deployment'] != '*NA']
dataset = dataset[dataset['Airbag Deployment'] != '*BL']
dataset = dataset[dataset['Airbag Deployment'] != '*ND']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
print dataset
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':15})
plt.show(cor_heat_map)'''

# Case 2 - Patient Position in Vehicle with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Patient Position in Vehicle', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*NA']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*ND']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
print dataset
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# Case 3 - Safety Equipment Issues with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Safet Equipment Issues', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Safet Equipment Issues'] != '*NA']
dataset = dataset[dataset['Safet Equipment Issues'] != '*ND']
dataset = dataset[dataset['Safet Equipment Issues'] != '*BL']
# dataset = dataset[dataset['Levels'] != '3']
# dataset = dataset[dataset['Levels'] != 'N']
print dataset
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# Case 4 - MV Speed with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','MV Speed', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['MV Speed'] != '*NA']
dataset = dataset[dataset['MV Speed'] != '*ND']
dataset = dataset[dataset['MV Speed'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# Case 5 - Fall Height with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Fall Height', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Fall Height'] != '*NA']
dataset = dataset[dataset['Fall Height'] != '*ND']
dataset = dataset[dataset['Fall Height'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# Case 6 - Transport Mode with trauma level --> Check once more
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Transport Mode', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
print dataset['Levels'].unique()
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# Case 7 - SBP with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','SBP', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['SBP'] != '*NA']
dataset = dataset[dataset['SBP'] != '*ND']
dataset = dataset[dataset['SBP'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
print dataset.groupby(['SBP','Levels']).size()'''

# Case 8 - Pulse with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Pulse', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Pulse'] != '*NA']
dataset = dataset[dataset['Pulse'] != '*ND']
dataset = dataset[dataset['Pulse'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
print dataset.groupby(['Pulse','Levels']).size()'''

# Case 9 - RR with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','RR', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['RR'] != '*NA']
dataset = dataset[dataset['RR'] != '*ND']
dataset = dataset[dataset['RR'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
print dataset.groupby(['RR','Levels']).size()'''

# Case 10 - Injury Severity Score with trauma level
'''dataset = pd.read_csv('data.csv')
dataset = dataset[['T1#','Injury Severity Score', 'Levels']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Injury Severity Score'] != '*NA']
dataset = dataset[dataset['Injury Severity Score'] != '*ND']
dataset = dataset[dataset['Injury Severity Score'] != '*BL']
dataset = dataset[dataset['Levels'] != '3']
dataset = dataset[dataset['Levels'] != 'N']
print dataset.groupby(['Injury Severity Score','Levels']).size()'''

# PRE WITH POST

# 1 - Report of physical abuse?
# 1.a - Trauma Type
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#', 'Report of physical abuse?', 'Trauma Type']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Report of physical abuse?'] != '*NA']
dataset = dataset[dataset['Report of physical abuse?'] != '*ND']
dataset = dataset[dataset['Report of physical abuse?'] != '*BL']
dataset = dataset[dataset['Trauma Type'] != '*NA']
dataset = dataset[dataset['Trauma Type'] != '*ND']
dataset = dataset[dataset['Trauma Type'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5})
plt.show(cor_heat_map)

# 1.b - Transport Mode
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#', 'Report of physical abuse?', 'Transport Mode']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Report of physical abuse?'] != '*NA']
dataset = dataset[dataset['Report of physical abuse?'] != '*ND']
dataset = dataset[dataset['Report of physical abuse?'] != '*BL']
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5})
plt.show(cor_heat_map)

# 1.c - Arrived From
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#', 'Report of physical abuse?', 'Arrived From']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Report of physical abuse?'] != '*NA']
dataset = dataset[dataset['Report of physical abuse?'] != '*ND']
dataset = dataset[dataset['Report of physical abuse?'] != '*BL']
dataset = dataset[dataset['Arrived From'] != '*NA']
dataset = dataset[dataset['Arrived From'] != '*ND']
dataset = dataset[dataset['Arrived From'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5})
plt.show(cor_heat_map)

# 2 - Airbag Deployment
# 2.a - Trauma Type
'''dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Airbag Deployment', 'Trauma Type']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Airbag Deployment'] != '*NA']
dataset = dataset[dataset['Airbag Deployment'] != '*ND']
dataset = dataset[dataset['Airbag Deployment'] != '*BL']
dataset = dataset[dataset['Trauma Type'] != '*NA']
dataset = dataset[dataset['Trauma Type'] != '*ND']
dataset = dataset[dataset['Trauma Type'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 2.b - Transport Mode
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Airbag Deployment', 'Transport Mode']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Airbag Deployment'] != '*NA']
dataset = dataset[dataset['Airbag Deployment'] != '*ND']
dataset = dataset[dataset['Airbag Deployment'] != '*BL']
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 2.c - Arrived From
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Airbag Deployment', 'Arrived From']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Airbag Deployment'] != '*NA']
dataset = dataset[dataset['Airbag Deployment'] != '*ND']
dataset = dataset[dataset['Airbag Deployment'] != '*BL']
dataset = dataset[dataset['Arrived From'] != '*NA']
dataset = dataset[dataset['Arrived From'] != '*ND']
dataset = dataset[dataset['Arrived From'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# 3 - Patient Position in Vehicle
# 3.a - Trauma Type
'''dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Patient Position in Vehicle', 'Trauma Type']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*NA']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*ND']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*BL']
dataset = dataset[dataset['Trauma Type'] != '*NA']
dataset = dataset[dataset['Trauma Type'] != '*ND']
dataset = dataset[dataset['Trauma Type'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 3.b - Transport Mode
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Patient Position in Vehicle', 'Transport Mode']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*NA']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*ND']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*BL']
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 3.c - Arrived From
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Patient Position in Vehicle', 'Arrived From']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*NA']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*ND']
dataset = dataset[dataset['Patient Position in Vehicle'] != '*BL']
dataset = dataset[dataset['Arrived From'] != '*NA']
dataset = dataset[dataset['Arrived From'] != '*ND']
dataset = dataset[dataset['Arrived From'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# 4 - Child Restraint
# 4.a - Trauma Type
'''dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Child Restraint', 'Trauma Type']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Child Restraint'] != '*NA']
dataset = dataset[dataset['Child Restraint'] != '*ND']
dataset = dataset[dataset['Child Restraint'] != '*BL']
dataset = dataset[dataset['Trauma Type'] != '*NA']
dataset = dataset[dataset['Trauma Type'] != '*ND']
dataset = dataset[dataset['Trauma Type'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 4.b - Transport Mode
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Child Restraint', 'Transport Mode']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Child Restraint'] != '*NA']
dataset = dataset[dataset['Child Restraint'] != '*ND']
dataset = dataset[dataset['Child Restraint'] != '*BL']
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 4.c - Arrived From
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Child Restraint', 'Arrived From']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Child Restraint'] != '*NA']
dataset = dataset[dataset['Child Restraint'] != '*ND']
dataset = dataset[dataset['Child Restraint'] != '*BL']
dataset = dataset[dataset['Arrived From'] != '*NA']
dataset = dataset[dataset['Arrived From'] != '*ND']
dataset = dataset[dataset['Arrived From'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# 5 - Final Outcome-Dead or Alive
# 5.a - Trauma Type
'''dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Final Outcome-Dead or Alive', 'Trauma Type']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*NA']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*ND']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*BL']
dataset = dataset[dataset['Trauma Type'] != '*NA']
dataset = dataset[dataset['Trauma Type'] != '*ND']
dataset = dataset[dataset['Trauma Type'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 5.b - Transport Mode
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Final Outcome-Dead or Alive', 'Transport Mode']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*NA']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*ND']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*BL']
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 5.c - Arrived From
dataset = pd.read_csv('newData.csv')
print dataset
dataset = dataset[['T1#','Final Outcome-Dead or Alive', 'Arrived From']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*NA']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*ND']
dataset = dataset[dataset['Final Outcome-Dead or Alive'] != '*BL']
dataset = dataset[dataset['Arrived From'] != '*NA']
dataset = dataset[dataset['Arrived From'] != '*ND']
dataset = dataset[dataset['Arrived From'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''

# 6 - Injury Severity Score
# 6.a - Trauma Type
'''dataset = pd.read_csv('newData.csv')
dataset = dataset[['T1#','Injury Severity Score', 'Trauma Type']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Injury Severity Score'] != '*NA']
dataset = dataset[dataset['Injury Severity Score'] != '*ND']
dataset = dataset[dataset['Injury Severity Score'] != '*BL']
dataset = dataset[dataset['Trauma Type'] != '*NA']
dataset = dataset[dataset['Trauma Type'] != '*ND']
dataset = dataset[dataset['Trauma Type'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 6.b - Transport Mode
dataset = pd.read_csv('newData.csv')
dataset = dataset[['T1#','Injury Severity Score', 'Transport Mode']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Injury Severity Score'] != '*NA']
dataset = dataset[dataset['Injury Severity Score'] != '*ND']
dataset = dataset[dataset['Injury Severity Score'] != '*BL']
dataset = dataset[dataset['Transport Mode'] != '*NA']
dataset = dataset[dataset['Transport Mode'] != '*ND']
dataset = dataset[dataset['Transport Mode'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)

# 6.c - Arrived From
dataset = pd.read_csv('newData.csv')
dataset = dataset[['T1#','Injury Severity Score', 'Arrived From']]
dataset = pd.DataFrame.drop_duplicates(dataset)
del dataset['T1#']
dataset = dataset[dataset['Injury Severity Score'] != '*NA']
dataset = dataset[dataset['Injury Severity Score'] != '*ND']
dataset = dataset[dataset['Injury Severity Score'] != '*BL']
dataset = dataset[dataset['Arrived From'] != '*NA']
dataset = dataset[dataset['Arrived From'] != '*ND']
dataset = dataset[dataset['Arrived From'] != '*BL']
d = pd.get_dummies(dataset)
d = pd.DataFrame.corr(d)
print d
sb.set(font_scale=1)
cor_heat_map = sb.heatmap(d, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':5})
plt.show(cor_heat_map)'''
