import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
from scipy.stats.stats import pearsonr

df = pd.read_csv('refined_data_todd.csv', header = None, error_bad_lines=False)
df.columns = ['T1',	'ED/Hosp Arrival Date',	'Age in Years',	'Gender',	'Levels',	'ICD-10 E-code',	'Trauma Type',	'Report of physical abuse',	'Injury Comments',	'Airbag Deployment',	'Patient Position in Vehicle',
              'Safet Equipment Issues',	'Child Restraint',	'MV Speed',	'Fall Height',	'Transport Type',	'Transport Mode',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'Resp Assistance',
              'RTS',	'Field GCS',	'Arrived From',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',
              'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',

              'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005']

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


#pre admission
gender_column = df.iloc[:,3].values
gender_column = le.fit_transform(gender_column)
genderlist = []

for i in gender_column:
    if i == 2 or i == 0:
        genderlist.append(i)
    else:
        genderlist.append(0.00000000000000000001)

df['Gender_Numerical'] = genderlist


#pre admission
RTS = df.iloc[:,22].values
RTS_List = column_NullHandler(RTS)
df['RTS_Refined'] = RTS_List


#pre admission

transport_type = df.iloc[:,15].values
transport_list = le.fit_transform(transport_type)
transport_type = []
for i in transport_list:
    if i == 0:
        transport_type.append(0.00000000000000000001)
    else:
        transport_type.append(i)

df['transport_type'] = transport_type


#pre admission
age = df.iloc[:,2].values
age_list = column_NullHandler(age)
df['age_list'] = age_list




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





