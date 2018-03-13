import pandas as pd

df = pd.read_csv('/Users/Pramod/Desktop/SER517/Original datasets/Copy of Stacked Trauma data.csv', header = None, error_bad_lines=False)
df.columns = ['T1#','Medical Record Number', 'ED/Hosp Arrival Date', 'Date of Birth', 'Age + Units', 'Age ', 'Units of Age' ,
              'Gender', 'Levels', 'Co-morbid 1', 'Co-morbid 2', 'ICD-10 E-code #1', 'Trauma Type', 'Report of physical abuse?',
              'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle', 'Safet Equipment Issues', 'Child Restraint',
              'MV Speed', 'Fall Height', 'Transport Mode', 'SBP', 'Pulse', 'RR', 'GCS', 'Intubated < Vitals?', 'Arrived From',
              'ED LOS (mins)', 'Dispositon from  ED', 'Final Outcome-Dead or Alive', 'Discharge Disposition', 'GCS Total', 'Total Days in ICU',
              'Admission Hosp LOS (days)', 'Treatment/Intervention', 'Total LOS (ED+Admit)', 'Time to 1st OR Visit (mins.)', 'Injury Severity Score',
              'AIS 2005 (Injury no 1)','AIS 2005 Body Part','AIS 2005 Severity','ICD10 Dx Code','Consulting Service','Procedure_Performed',
              'Location', 'Service', 'Anesthesia Start Time', 'Time to Proc (ED Arrival) Min']

df = df[['Levels', 'Treatment/Intervention', 'Consulting Service', 'Procedure_Performed', 'Service']]

#print df
# The size of dataframe is 14836 rows x 5 columns

#***********************************************************************
#Procedure Performed Dataframe
df_Procedure_Performed = df[['Levels', 'Procedure_Performed']]
print "Number of rows:"
print len(df_Procedure_Performed)

#Dropping all the rows with null values
list1 = ['Levels', 'Procedure_Performed']
for x in list1:
    df_Procedure_Performed = df_Procedure_Performed[pd.notnull(df[x])]
print "Number of rows after dropping null values:"
print len(df_Procedure_Performed)

df_List_of_Procedures = df_Procedure_Performed[['Procedure_Performed']]
print "Number of rows of just Procedures:"
print len (df_List_of_Procedures)

df_List_of_Procedures = df_List_of_Procedures.drop_duplicates()
print "Number of rows of all Procedures without repetition:"
print len (df_List_of_Procedures)

list_of_procedures_unique = df_List_of_Procedures.Procedure_Performed.tolist()
print "The length of unique procedures list:"
print len(list_of_procedures_unique)

list_of_procedures = df_Procedure_Performed.Procedure_Performed.tolist()
list_of_levels = df_Procedure_Performed.Levels.tolist()

print "The length of all procedures list:"
print len(list_of_procedures)

print "The length of all levels list:"
print len(list_of_levels)




