import pandas as pd
import math
import numpy as np
from sklearn.decomposition import PCA


df = pd.read_csv('Pramod_PCA_data.csv', header = None, error_bad_lines=False)
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type', 'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',    'Field Shock Index',   'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score',   'AIS 2005']


def average(list1):
    avgList = []
    count = 0;
    for i in list1:
        if i != "":
            avgList.append(i)
            count = count + 1
    return sum(avgList)/count

def replaceNan(list2):
    nanlist = []
    for i in list2:
        x = float(i)
        if not math.isnan(x):
            nanlist.append(x)
        else:
            nanlist.append(0.0)
    return nanlist

# Refining the precondition Data
#pre admission (Field_SBP)
Field_SBP = df.iloc[:,17].values
Field_SBP_withoutNan = replaceNan(Field_SBP)
avg_SBP = average(Field_SBP_withoutNan)

SBP_List = []
for i in Field_SBP:
    x = float(i)
    if not math.isnan(x):
        SBP_List.append(x)
    else:
        SBP_List.append(avg_SBP)

#print SBP_List

#pre admission (Field_HR)
Field_HR = df.iloc[:,18].values
Field_HR_withoutNan = replaceNan(Field_HR)
avg_HR = average(Field_HR_withoutNan)

HR_List = []
for i in Field_HR:
    x = float(i)
    if not math.isnan(x):
        HR_List.append(x)
    else:
        HR_List.append(avg_HR)
# print HR_List

#pre admission (Field_RR)
Field_RR = df.iloc[:,18].values
Field_RR_withoutNan = replaceNan(Field_RR)
avg_RR = average(Field_RR_withoutNan)

RR_List = []
for i in Field_RR:
    x = float(i)
    if not math.isnan(x):
        RR_List.append(x)
    else:
        RR_List.append(avg_RR)
# print RR_List

#pre admission (Field_RR)
Field_Shock = df.iloc[:,18].values
Field_Shock_withoutNan = replaceNan(Field_Shock)
avg_Shock = average(Field_Shock_withoutNan)

Shock_List = []
for i in Field_Shock:
    x = float(i)
    if not math.isnan(x):
        Shock_List.append(x)
    else:
        Shock_List.append(avg_Shock)
#print Shock_List

X = pd.DataFrame(
    {'Respiratory_Rate': RR_List,
     'Heart_Rate': HR_List,
     'Field_Shock': Shock_List,
     'SBP': SBP_List
    })

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
#print X
#print X

pca = PCA(n_components=2)
pca.fit(X)

#This tells us the amount of information retained in each Principal Component
variance= pca.explained_variance_ratio_

print "The variance retained in Principal Components 1 and 2 are: "
print variance

# principalDf = pd.DataFrame(data = X
#              , columns = ['principal component 1', 'principal component 2'])
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)

#Here we cumulatively add the prinipal component scores and make sure that it reaches a 100
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print var1

import matplotlib.pyplot as plt
plt.plot(var1)
plt.show()
