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

#Taking the rows which do not contain the words FALL, MOTOR or CAR (na = False means that it igoners the rows with NaN values)
df_assault_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains(r'^(?:(?!FALL|MOTOR|CAR).)*$', na = False) ]

#df_assault_icd9 =  df['A'].str.contains(r'^(?:(?!ASSAULT|World).)*$')
print "Number of rows:"
print len(df_assault_icd9)
print ""

#Taking the rows which do not contain the words FALL, MOTOR or CAR (na = False means that it igoners the rows with NaN values)
df_assault_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains(r'^(?:(?!FALL|MOTOR|CAR).)*$', na = False) ]
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
#print result


#Only considering the columns we will require
result = result[['Levels', 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']]

# Taking only the rows with levels 1 and 2 trauma levels
result = result.loc[result['Levels'].isin(['1', '2'])]

#Replacing Male with 1 and female with 2
result['Gender'] = result['Gender'].replace(['M', 'F'], value = ['1', '2'])

#*************************************************************************
list1 = [result['Field SBP'], result['Field HR'], result['Field RR'], result['RTS'], result['Field GCS']]

def create_list(x):
    return x.tolist()

# list_dummy = create_list(result['Field SBP'])

def average(list):
    avgList = []
    count = 0;
    for i in list:
        if i != '*NA' and i != '*ND' and i != '*BL':
            avgList.append(float(i))
            count = count + 1
    return (sum(avgList)/count)

# print "the average of dummy list is"
# print average(list_dummy)

list_names = ['Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']

list_avg_values = []

for i in range(0, len(list1)):
    print list_names[i]
    x = (average(create_list(list1[i])))
    print x
    list_avg_values.append(x)

#*************************************************************************

#Replacing the NA, ND, and Bl rows with a very small value
list2 = ['Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']
for x in range(0, len(list2)):
    result[list_names[x]] = result[list_names[x]].replace(['*NA', '*ND', '*BL'], value=[list_avg_values[x], list_avg_values[x], list_avg_values[x]])

print result
#*************************************************************************


