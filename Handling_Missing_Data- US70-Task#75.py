import pandas as pd
import sys
from pandas import ExcelWriter


df = pd.read_excel('NewFallDataset.xlsx', header = None)

df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code','ICD-9 E-code',  'Trauma Type', 'Report of physical abuse',
              'Injury Comments', 'Airbag Deployment',
              'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',
              'Field HR', 'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)','Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Hospital Disposition',   'Injury Severity Score', 'ICD 9 Dx (before 2016)','ICD 10 Dx (after 1/2016)','AIS 2005']


list = ['T1',  'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code','ICD-9 E-code',  'Trauma Type', 'Report of physical abuse',
              'Injury Comments', 'Airbag Deployment',
              'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',
              'Field HR', 'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)','Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Hospital Disposition',   'Injury Severity Score', 'ICD 9 Dx (before 2016)','ICD 10 Dx (after 1/2016)','AIS 2005']


list = ['Levels', 'Age in Years', 'Fall Height', 'Gender','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']

for x in list:
    df = df[pd.notnull(df[x])]

def findAverage(list):
    sum = 0
    for i in range(0,len(list)):
        if list[i] == "*NA" or list[i] == "*ND" or list[i] == "*BL":
            list[i] = sys.maxint

    newlist = []
    for i in list:
        if i == sys.maxint:
            continue
        else:
            newlist.append(i)
    newlist.pop(0)

    for i in newlist:
        sum = sum + i
    average = sum/len(list)
    return average



FallHeightList = df['Fall Height']
FieldSBPList = df['Field SBP']
FieldHR = df['Field HR']
FieldRR = df['Field RR']
RTS = df['RTS']
FieldGCS = df['Field GCS']
AgeInYears = df['Age in Years']
EDLOS = df['ED LOS (mins)']
MVSpeedList = df['MV Speed']



averageFallHeight = findAverage(FallHeightList)

avergaeFieldSBP = findAverage(FieldSBPList)

avgFieldHR = findAverage(FieldHR)

avgFieldRR = findAverage(FieldRR)

avgFieldRTS = findAverage(RTS)

avgFieldGCS = findAverage(FieldGCS)

avgAge = findAverage(AgeInYears)

AvgEDLOS = findAverage(EDLOS)

avgMVSpeed = findAverage(MVSpeedList)



df['Fall Height'] = df['Fall Height'].replace([sys.maxint], value = [averageFallHeight])

df['Field SBP'] = df['Field SBP'].replace([sys.maxint], value = [avergaeFieldSBP])

df['Field HR'] = df['Field HR'].replace([sys.maxint], value = [avgFieldHR])

df['Field RR'] = df['Field RR'].replace([sys.maxint], value = [avgFieldRR])

df['RTS'] = df['RTS'].replace([sys.maxint], value = [avgFieldRTS])

df['Field GCS'] = df['Field GCS'].replace([sys.maxint], value = [avgFieldGCS])

df['Age in Years'] = df['Age in Years'].replace([sys.maxint], value = [avgAge])

df['ED LOS (mins)'] = df['ED LOS (mins)'].replace([sys.maxint], value = [AvgEDLOS])

df['Report of physical abuse'] = df['Report of physical abuse'].replace(['*BL'], value = ['0'])

df['Trauma Type'] = df['Trauma Type'].replace(['Blunt','Penetrating'], value = ['1','0'])

df['Airbag Deployment'] = df['Airbag Deployment'].replace(['*NA'], value = ['0'])

df['Safet Equipment Issuest'] = df['Safet Equipment Issues'].replace(['*NA'], value = ['0'])

df['Child Restraint'] = df['Child Restraint'].replace(['*NA'], value = ['0'])

df['MV Speed'] = df['MV Speed'].replace(['*NA',sys.maxint], value = ['0',avgMVSpeed])



writer = ExcelWriter('Sprint5US_70DataSet.xlsx')
df.to_excel(writer,'Sheet5')
writer.save()
