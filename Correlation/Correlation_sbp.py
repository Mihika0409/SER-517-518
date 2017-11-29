import pandas as pd
import math
import scipy

df = pd.read_csv('/Users/Pramod/Desktop/SER517/Correlation/Trauma2_removeND_NA_BL.csv', header = None, error_bad_lines=False)
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

#print HR_List

#Refining the Post Condition Data

#post admission (ED SBP)
ED_SBP = df.iloc[:,27].values
ED_SBP_withoutNan = replaceNan(ED_SBP)
avg_SBP1 = average(ED_SBP_withoutNan)

SBPED_List = []
for i in ED_SBP:
    x = float(i)
    if not math.isnan(x):
        SBPED_List.append(x)
    else:
        SBPED_List.append(avg_SBP1)

#print SBPED_List

#post admission (ED HR)
ED_HR = df.iloc[:,28].values
ED_HR_withoutNan = replaceNan(ED_HR)
avg_HR1 = average(ED_HR_withoutNan)

HRED_List = []
for i in ED_HR:
    x = float(i)
    if not math.isnan(x):
        HRED_List.append(x)
    else:
        HRED_List.append(avg_HR1)

#print HRED_List

#post admission (ED RR)
ED_RR = df.iloc[:,29].values
ED_RR_withoutNan = replaceNan(ED_RR)
avg_RR1 = average(ED_RR_withoutNan)

RRED_List = []
for i in ED_RR:
    x = float(i)
    if not math.isnan(x):
        RRED_List.append(x)
    else:
        RRED_List.append(avg_RR1)

#print RRED_List

#post admission (ED GCS)
ED_GCS = df.iloc[:,30].values
ED_GCS_withoutNan = replaceNan(ED_GCS)
avg_GCS1 = average(ED_GCS_withoutNan)

GCSED_List = []
for i in ED_GCS:
    x = float(i)
    if not math.isnan(x):
        GCSED_List.append(x)
    else:
        GCSED_List.append(avg_GCS1)

#print GCSED_List

from scipy import stats
print "The Spearman's Coefficient's are:"
print "Field_SBP and ED_SBP:", + scipy.stats.stats.spearmanr(SBP_List, SBPED_List)[0]
print "Field_SBP and ED_HR:", + scipy.stats.stats.spearmanr(SBP_List, HRED_List)[0]
print "Field_SBP and ED_RR:", + scipy.stats.stats.spearmanr(SBP_List, RRED_List)[0]
print "Field_SBP and ED_GCS:", + scipy.stats.stats.spearmanr(SBP_List, GCSED_List)[0]
print""
print "Field_HR and ED_SBP:", + scipy.stats.stats.spearmanr(HR_List, SBPED_List)[0]
print "Field_HR and ED_SBP:", + scipy.stats.stats.spearmanr(HR_List, HRED_List)[0]
print "Field_HR and ED_SBP:", + scipy.stats.stats.spearmanr(HR_List, RRED_List)[0]
print "Field_HR and ED_SBP:", + scipy.stats.stats.spearmanr(HR_List, GCSED_List)[0]
