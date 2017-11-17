import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
from scipy.stats.stats import pearsonr

df = pd.read_csv('Trauma2_removeND_NA_BL.csv', header = None, error_bad_lines=False)
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type', 'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',    'Field Shock Index',   'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score',   'AIS 2005']

#[5 rows x 42 columns]
#print df.head()


le = LabelEncoder()

def column_NullHandler(columnlist):
    newlist = []
    for i in columnlist:
        x = float(i)
        if not math.isnan(x):
            newlist.append(x)
        else:
            newlist.append(0.0000000000000000001)
    return newlist


#pre admission (Field_SBP)
Field_SBP = df.iloc[:,17].values
SBP_List = []
for i in Field_SBP:
    x = float(i)
    if not math.isnan(x):
        SBP_List.append(x)
    else:
        SBP_List.append(0.0000000000000000001)

#print SBP_List


df['SBP'] = SBP_List

#[5 rows x 43 columns]
#print df.head()


#pre admission (Field_HR)
Field_HR = df.iloc[:,18].values
HR_List = []
for i in Field_HR:
    x = float(i)
    if not math.isnan(x):
        HR_List.append(x)
    else:
        HR_List.append(0.0000000000000000001)

df['HR'] = HR_List

#[5 rows x 44 columns]
#print df.head()


#----------------------------------------------------
# #pre admission (Field_Shock)
# Field_Shock = df.iloc[:,19].values
# Shock_List = []
# for i in Field_Shock:
#     x = float(i)
#     if not math.isnan(x):
#         Shock_List.append(x)
#     elif x == '#VALUE!':
#         Shock_List.append(0.0000000000000000001)
#     else:
#         Shock_List.append(0.0000000000000000001)
#
# df['Shock'] = Shock_List
#
# #[5 rows x 44 columns]
# print df.head()
#----------------------------------------------------

#post admission 1
report_physical_abuse = df.iloc[:,7].values
report_physical_abuse = le.fit_transform(report_physical_abuse)


physicalAbuseList  = []
for i in report_physical_abuse:
    if i == 1 or i == 2:
        physicalAbuseList.append(i)
    else:
        physicalAbuseList.append(0.00000000000000000001)
df['report_physical_abuse_numerical'] = physicalAbuseList


#postadmission 2


airbag_deployment = df.iloc[:,9].values
airbag_deployment = le.fit_transform(airbag_deployment)

airbagList = []
for i in airbag_deployment:
    if i == 0:
        airbagList.append(0.00000000000000000001)
    else:
        airbagList.append(i)
df['airbagdeploy'] = airbagList



#post admission 2
passenger_seat = df.iloc[:,9].values
passenger_seat = le.fit_transform(passenger_seat)

passenger_list = []
for i in airbag_deployment:
    if i == 0:
        passenger_list.append(0.00000000000000000001)
    else:
        passenger_list.append(i)
df['passenger_seat'] = passenger_list

#print df['Gender_Numerical'].corr(df['passenger_seat'])


#post admission
total_vent_days = df.iloc[:,31].values
newlist = column_NullHandler(total_vent_days)
df['total_vent_days'] = newlist




#post admission
total_ICU_Days = df.iloc[:,32].values
ICU_List = column_NullHandler(total_ICU_Days)
df['total_ICU_days'] = ICU_List



#post admission
admission_hosp_days = df.iloc[:,33].values
admission_hospList = column_NullHandler(admission_hosp_days)
df['adm_hosp_days'] = admission_hospList


#post admission
dead_alive = df.iloc[:,38].values
dead_alive = le.fit_transform(dead_alive)
doa_list = []
for i in dead_alive:
    if i == 0:
        doa_list.append(0.00000000000000000001)
    else:
        doa_list.append(i)
df['dead_or_alive'] = doa_list



#post admission
injury_sev_score = df.iloc[:,40].values
injury_sev_list = column_NullHandler(injury_sev_score)
df['injury_sev_list'] = injury_sev_list



print "==============================================="
print "correlation between SBP and the post factors"
#
print "SBP and Report Physical Abuse", + df['SBP'].corr(df['report_physical_abuse_numerical'])
print "SBP and AirbagDeploy", + df['SBP'].corr(df['airbagdeploy'])
print "SBP and ED LOS", + df['SBP'].corr(df['ED LOS (mins)'])
print "SBP and ED SBP", + df['SBP'].corr(df['ED SBP'])
print "SBP and ED HR", + df['SBP'].corr(df['ED HR'])
print "SBP and resp rate", +df['SBP'].corr(df['ED RR'])
print "SBP and GCS", +df['SBP'].corr(df['ED GCS'])
print "SBP and no of vent days", +df['SBP'].corr(df['total_vent_days'])
print "SBP and no of ICU days", +df['SBP'].corr(df['total_ICU_days'])
print "SBP and no of adm hosp days", +df['SBP'].corr(df['adm_hosp_days'])
print "SBP and outcome_deadOrAlive", +df['SBP'].corr(df['dead_or_alive'])
print "SBP and outcome_deadOrAlive", +df['SBP'].corr(df['injury_sev_list'])


print "==============================================="
print "correlation between HR and the post factors"
#
print "HR and Report Physical Abuse", + df['HR'].corr(df['report_physical_abuse_numerical'])
print "HR and AirbagDeploy", + df['HR'].corr(df['airbagdeploy'])
print "HR and ED LOS", + df['HR'].corr(df['ED LOS (mins)'])
print "HR and ED SBP", + df['HR'].corr(df['ED SBP'])
print "HR and ED HR", + df['HR'].corr(df['ED HR'])
print "HR and resp rate", +df['HR'].corr(df['ED RR'])
print "HR and GCS", +df['HR'].corr(df['ED GCS'])
print "HR and no of vent days", +df['HR'].corr(df['total_vent_days'])
print "HR and no of ICU days", +df['SBP'].corr(df['total_ICU_days'])
print "HR and no of adm hosp days", +df['HR'].corr(df['adm_hosp_days'])
print "HR and outcome_deadOrAlive", +df['HR'].corr(df['dead_or_alive'])
print "HR and outcome_deadOrAlive", +df['HR'].corr(df['injury_sev_list'])


