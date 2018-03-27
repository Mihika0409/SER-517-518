import pandas as pd

df = pd.read_csv('/Users/Pramod/Desktop/SER517/Original datasets/FullData.csv', header = None, error_bad_lines=False)
df.columns = ['T1#', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code (after 1/2016)', 'ICD-9 E-code (before 2016)',
              'Trauma Type', 'Report of abuse? (started 2014)', 'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle',
              'Safety Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type', 'Transport Mode', 'Field SBP',
              'Field HR', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED Disposition', 'ED SBP',
              'ED HR','ED RR', 'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive', 'Hospital Disposition',
              'Injury Severity Score', 'ICD 9 Dx (before 2016)', 'ICD 10 Dx (after 1/2016)', 'AIS 2005']

#********************************************************************************
#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd9_fall = df[df['ICD-9 E-code (before 2016)'].str.contains("FALL", na = False) ]
print "Number of rows:"
print len(df_assault_icd9_fall)
print ""

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd10_fall = df[df['ICD-10 E-code (after 1/2016)'].str.contains("FALL", na = False) ]
print "Number of rows:"
print len(df_assault_icd10_fall)
print ""
#********************************************************************************

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains("ASSAULT", na = False) ]
print "Number of rows:"
print len(df_assault_icd9)
print ""

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains("ASSAULT", na = False) ]
print "Number of rows:"
print len(df_assault_icd10)
print ""

print "The dataframe is: "
#print df_assault_icd10

#Combining data of both icd 9 and icd 10 codes
frames = [df_assault_icd9, df_assault_icd10]
result = pd.concat(frames)
print "Number of rows in complete dataframe:"
print len(result)
#Printing out the resultant dataframe
print result