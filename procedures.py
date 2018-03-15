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

performance_procedure_performed = []
objects_procedure_performed = []
for k, v in level1_procedure_dict.iteritems():
        print k, v
        objects_procedure_performed.append(k)
        performance_procedure_performed.append(v)

performance_2_procedure_performed = []
objects_2_procedure_performed = []
for k, v in level2_procedure_dict.iteritems():
        print k, v
        objects_2_procedure_performed.append(k)
        performance_2_procedure_performed.append(v)

print "length of object 1:"
print len(objects_procedure_performed)
print len(performance_procedure_performed)

print "length of object 2:"
print len(objects_2_procedure_performed)


#***********************************************************************
print ""
print "The Visualization for Service Performed"

#Service Dataframe
df_Service = df[['Levels', 'Service']]
print "Number of rows:"
print len(df_Service)

#Dropping all the rows with null values
list1 = ['Levels', 'Service']
for x in list1:
    df_Service = df_Service[pd.notnull(df[x])]
print "Number of rows after dropping null values:"
print len(df_Service)


df_List_Services = df_Service[['Service']]
print "Number of rows of just services:"
print len (df_List_Services)

df_List_Services = df_List_Services.drop_duplicates()
print "Number of rows of all services without repetition:"
print len (df_List_Services)

list_of_services_unique = df_List_Services.Service.tolist()
print "The length of unique services list:"
print len(list_of_services_unique)

list_of_services = df_Service.Service.tolist()
list_of_levels = df_Service.Levels.tolist()

print "The length of all services list:"
print len(list_of_services)

print "The length of all levels list:"
print len(list_of_levels)


level1_service_dict = {}
level2_service_dict = {}

for i in range(0, len(list_of_services_unique)):
    service_level1 = 0;
    service_level2 = 0;
    for j in range(0, len(list_of_services)):
        if list_of_services[j] == list_of_services_unique[i] and list_of_levels[j] == '1':
            service_level1 = service_level1 + 1

        if list_of_services[j] == list_of_services_unique[i] and list_of_levels[j] == '2':
            service_level2 = service_level2 + 1

    level1_service_dict[list_of_services_unique[i]] = service_level1
    level2_service_dict[list_of_services_unique[i]] = service_level2

performance_service = []
objects_service = []
for k, v in level1_service_dict.iteritems():
        print k, v
        objects_service.append(k)
        performance_service.append(v)

performance_2_service = []
objects_2_service = []
for k, v in level2_service_dict.iteritems():
        print k, v
        objects_2_service.append(k)
        performance_2_service.append(v)

print "length of object 1 of services:"
print len(objects_service)
print len(performance_service)

print "length of object 2 of services:"
print len(objects_2_service)


def graph(performance1, performance2, objects1, name):
    import numpy as np
    import matplotlib.pyplot as plt
    # data to plot
    n_groups = len(objects1)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, performance1, bar_width,
                     alpha=opacity,
                     color='b',
                     label='level1')

    rects2 = plt.bar(index + bar_width, performance2, bar_width,
                     alpha=opacity,
                     color='g',
                     label='level2')

    plt.xlabel(name)
    plt.ylabel('Count of levels')
    plt.title('Count of different parameters')
    plt.xticks(index + bar_width, list_of_procedures_unique)
    plt.legend()

    # plt.tight_layout()
    plt.show()

print "The graph for procedure performed is: "
graph(performance_procedure_performed, performance_2_procedure_performed, objects_procedure_performed, "Procedure Performed")

print "The graph for services: "
graph(performance_service, performance_2_service, objects_service, "Service")
