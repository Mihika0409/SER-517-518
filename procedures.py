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



level1_procedure_dict = {}
level2_procedure_dict = {}

for i in range(0, len(list_of_procedures_unique)):
    procedure_level1 = 0;
    procedure_level2 = 0;
    for j in range(0, len(list_of_procedures)):
        if list_of_procedures[j] == list_of_procedures_unique[i] and list_of_levels[j] == '1':
            procedure_level1 = procedure_level1 + 1

        if list_of_procedures[j] == list_of_procedures_unique[i] and list_of_levels[j] == '2':
            procedure_level2 = procedure_level2 + 1

    level1_procedure_dict[list_of_procedures_unique[i]] = procedure_level1
    level2_procedure_dict[list_of_procedures_unique[i]] = procedure_level2

performance = []
objects = []
for k, v in level1_procedure_dict.iteritems():
        print k, v
        objects.append(k)
        performance.append(v)

performance_2 = []
objects_2 = []
for k, v in level1_procedure_dict.iteritems():
        print k, v
        objects_2.append(k)
        performance_2.append(v)

print "length of object 1:"
print len(objects)
print len(performance)

print "length of object 2:"
print len(objects_2)


#Visualizing the data
# import matplotlib.pyplot as plt;
#
# plt.rcdefaults()
# import numpy as np
# import matplotlib.pyplot as plt
#
# # objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# y_pos = np.arange(0, 20 * len(objects), 20)
# # performance = [10, 8, 6, 4, 2, 1]
#
# plt.bar(y_pos, performance, align='edge', alpha=0.5, width = 1.0)
# plt.xticks(y_pos, objects)
# plt.ylabel('level 1 cases')
# plt.title('Count of level 1 for Different procedures:')
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = len(objects)
# means_frank = (90, 55, 40, 65)
# means_guido = (85, 62, 54, 20)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, performance, bar_width,
                 alpha=opacity,
                 color='b',
                 label='level1')

rects2 = plt.bar(index + bar_width, performance_2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='level2')

plt.xlabel('Procedure')
plt.ylabel('Count of levels')
plt.title('Count by procedure')
plt.xticks(index + bar_width, list_of_procedures_unique)
plt.legend()

# plt.tight_layout()
plt.show()


