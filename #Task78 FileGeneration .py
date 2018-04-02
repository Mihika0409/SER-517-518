import pandas as pd
from pyxlsb import open_workbook as open_xlsb
from pandas import ExcelWriter

df = []

with open_xlsb('FullData.xlsb') as wb:
    with wb.get_sheet(1) as sheet:
        for row in sheet.rows():
            df.append([item.v for item in row])

df = pd.DataFrame(df[1:], columns=df[0])

df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code','ICD-9 E-code',  'Trauma Type', 'Report of physical abuse',
              'Injury Comments', 'Airbag Deployment',
              'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',
              'Field HR', 'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)','Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Hospital Disposition',   'Injury Severity Score', 'ICD 9 Dx (before 2016)','ICD 10 Dx (after 1/2016)','AIS 2005']


df = df[pd.notnull(df['T1'])]


df = df[df['ICD-9 E-code'].str.contains("FALL", na = False)]

df = df.loc[df['Levels'].isin(['1', '2'])]


writer = ExcelWriter('NewFallDataset.xlsx')
df.to_excel(writer,'Sheet5')
writer.save()



