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
#Taking the rows with injury comments which contain the FALL critertia
df_fall_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains("FALL", na = False) ]
print "Number of rows:"
print len(df_fall_icd9)
print ""

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_fall_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains("FALL", na = False) ]
print "Number of rows:"
print len(df_fall_icd10)
print ""

#Combining data of both icd 9 and icd 10 codes
frames = [df_fall_icd9, df_fall_icd10]
result_fall = pd.concat(frames)
print "Number of rows in complete dataframe:"
length_fall = len(result_fall)
print length_fall
#Printing out the resultant dataframe
#print result_fall

#Converting fall dataframe to csv file
result_fall.to_csv('Fall_trauma_newdata.csv', encoding='utf-8', index=False)

#********************************************************************************
#Taking the rows with injury comments which contain the MOTOR vehicle critertia
df_motor_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains("MOTOR|CAR|TRAFFIC", na = False) ]
print "Number of rows:"
print len(df_motor_icd9)
print ""

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_motor_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains("MOTOR|CAR", na = False) ]
print "Number of rows:"
print len(df_motor_icd10)
print ""

#Combining data of both icd 9 and icd 10 codes
frames = [df_motor_icd9, df_motor_icd10]
result_motor = pd.concat(frames)
print "Number of rows in complete dataframe:"
length_motor = len(result_motor)
print length_motor
#Printing out the resultant dataframe
#print result_motor

#********************************************************************************
#Taking the rows with injury comments which contain the rest critertia

#Taking the rows which do not contain the words FALL, MOTOR or CAR (na = False means that it igoners the rows with NaN values)
df_rest_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains(r'^(?:(?!FALL|MOTOR|CAR|ASSAULT).)*$', na = False) ]

#df_assault_icd9 =  df['A'].str.contains(r'^(?:(?!ASSAULT|World).)*$')
print "Number of rows:"
print len(df_rest_icd9)
print ""

#Taking the rows which do not contain the words FALL, MOTOR or CAR (na = False means that it igoners the rows with NaN values)
df_rest_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains(r'^(?:(?!FALL|MOTOR|CAR|ASSAULT).)*$', na = False) ]
print "Number of rows:"
print len(df_rest_icd10)
print ""

print "The dataframe is: "
#print df_assault_icd10

#Combining data of both icd 9 and icd 10 codes
frames = [df_rest_icd9, df_rest_icd10]
result_rest = pd.concat(frames)
print "Number of rows in complete dataframe:"
length_rest = len(result_rest)
print length_rest
#Printing out the resultant dataframe
#print result

#********************************************************************************
#Taking the rows with injury comments which contain the ASSAULT critertia
df_assault_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains("ASSAULT", na = False) ]
print "Number of rows:"
print len(df_assault_icd9)
print ""

#Taking the rows with injury comments which contain the ASSAULT critertia
df_assault_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains("ASSAULT", na = False) ]
print "Number of rows:"
print len(df_assault_icd10)
print ""

#Combining data of both icd 9 and icd 10 codes
frames = [df_assault_icd9, df_assault_icd10]
result_assault = pd.concat(frames)
print "Number of rows in complete dataframe:"
length_assault = len(result_assault)
print length_assault
#Printing out the resultant dataframe
#print result_fall

#********************************************************************************
#Plotting Pie Chart

import matplotlib.pyplot as plt

# Data to plot
labels = 'Fall', 'Motor Vehicle Accidents', 'Rest of the data', 'Assault'
sizes = [length_fall, length_motor, length_rest, length_assault]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1, 0.1, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode = explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()