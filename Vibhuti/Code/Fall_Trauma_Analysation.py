import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats.stats import pearsonr

df = pd.read_csv('fall_analysing_data.csv', header = None, error_bad_lines=False)
df.columns = ['T1',	'ED/Hosp Arrival Date',	'Age in Years',	'Gender',	'Levels',	'ICD-10 E-code',	'Trauma Type',	'Report of physical abuse',	'Injury Comments',	'Airbag Deployment',	'Patient Position in Vehicle',
              'Safet Equipment Issues',	'Child Restraint',	'MV Speed',	'Fall Height',	'Transport Type',	'Transport Mode',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'Resp Assistance',
              'RTS',	'Field GCS',	'Arrived From',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',
              'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',

              'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005']
#print df.head()

trauma_level = df.iloc[:,4]
trauma_type = df.iloc[:,6]

penetration_count = 0
blunt_count = 0

for i in range(0,len(trauma_type)):
    if trauma_type[i] == 'penetratin':
        penetration_count += 1
#print penetration_count

for i in range(0, len(trauma_type)):
    if trauma_type[i] == 'Blunt':
        blunt_count += 1

#print blunt_count

count_level1_penetration  = 0

for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'penetratin' and trauma_level[i] == '1':
        count_level1_penetration += 1

count_level2_penetration = 0
for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'penetratin' and trauma_level[i] == '2':
        count_level2_penetration += 1

count_level3_penetration = 0
for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'penetratin' and trauma_level[i] == '3':
        count_level3_penetration += 1


count_levelN_penetration = 0
for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'penetratin' and trauma_level[i] == '4':
        count_level3_penetration += 1


#blunt
count_level1_blunt = 0
count_level2_blunt = 0
count_level3_blunt = 0
count_levelN_blunt = 0


for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'Blunt' and trauma_level[i] == '1':
        count_level1_blunt += 1

count_level2_penetration = 0
for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'Blunt' and trauma_level[i] == '2':
        count_level2_blunt += 1

count_level3_penetration = 0
for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'Blunt' and trauma_level[i] == '3':
        count_level3_blunt += 1


count_levelN_penetration = 0
for i in range(0,len(trauma_level)):
    if trauma_type[i] == 'Blunt' and trauma_level[i] == '4':
        count_level3_blunt += 1


print "====="
print count_level1_blunt
print count_level2_blunt
print count_level3_blunt
print count_levelN_blunt




age = df.iloc[:,2]
fall_height = df.iloc[:,14]
trauma_level = df.iloc[:,4]


#for fall height < 6

fallheight_level1 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) <= 6 and int(trauma_level[i]) == 1:
        fallheight_level1 += 1




fallheight_level2 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) <= 6 and int(trauma_level[i]) == 2:
        fallheight_level2 += 1


fallheight_level3 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) <= 6 and int(trauma_level[i]) == 3:
        fallheight_level3 += 1




fallheight_levelN = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) <= 6 and int(trauma_level[i]) == 4:
        fallheight_levelN = fallheight_levelN + 1






print fallheight_level1
print fallheight_level2
print fallheight_level3
print fallheight_levelN


#for fall height > 6 and < 15


fallheight15_level1 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 6 and int(fall_height[i]) <= 15 and int(trauma_level[i]) == 1:
        fallheight15_level1 += 1




fallheight15_level2 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 6 and int(fall_height[i]) <= 15 and int(trauma_level[i]) == 2:
        fallheight15_level2 += 1


fallheight15_level3 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 6 and int(fall_height[i]) <= 15 and int(trauma_level[i]) == 3:
        fallheight15_level3 += 1




fallheight15_levelN = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 6 and int(fall_height[i]) <= 15 and int(trauma_level[i]) == 4:
        fallheight15_levelN += 1

print fallheight15_level1
print fallheight15_level2
print fallheight15_level3
print fallheight15_levelN


#for fall height > 15

fallheight_greaterThan15_level1 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 15 and int(trauma_level[i]) == 1:
        fallheight_greaterThan15_level1 += 1




fallheight_greaterThan15_level2 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 15 and int(trauma_level[i]) == 2:
        fallheight_greaterThan15_level2 += 1


fallheight_greaterThan15_level3 = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) >15 and int(trauma_level[i]) == 3:
        fallheight_greaterThan15_level3 += 1




fallheight_greaterThan15_levelN = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 15 and int(trauma_level[i]) == 4:
        fallheight_greaterThan15_levelN += 1


print fallheight_greaterThan15_level1
print fallheight_greaterThan15_level2
print fallheight_greaterThan15_level3
print fallheight_greaterThan15_levelN



#ED GCS AND trauma level

ED_GCS = df.iloc[:,30]

trauma_level = df.iloc[:,4]

coma_State = [3,4,5,6,7,8]
moderate_diablity = [9,10,11,12]
good_condition= [13,14,15]


#for coma state
coma_state_level1_count = 0
for i in range(0,len(ED_GCS)):
    if int(ED_GCS[i]) >= 3 and int(ED_GCS[i]) <= 8 and int(trauma_level[i]) == 1:
        coma_state_level1_count += 1



coma_state_level2_count = 0
for i in range(0,len(ED_GCS)):
    if int(ED_GCS[i]) >= 3 and int(ED_GCS[i]) <= 8 and int(trauma_level[i]) == 2:
        coma_state_level2_count += 1


coma_state_level3_count = 0
for i in range(0, len(ED_GCS)):
    if int(ED_GCS[i]) >= 3 and int(ED_GCS[i]) <= 8 and int(trauma_level[i]) == 3:
        coma_state_level3_count += 1

coma_state_levelN_count = 0
for i in range(0, len(ED_GCS)):
    if int(ED_GCS[i]) >= 3 and int(ED_GCS[i]) <= 8 and int(trauma_level[i]) == 4:
        coma_state_levelN_count += 1

print "For coma state ed_gcs between 3 - 8 and trauma level 1",coma_state_level1_count
print "For coma state ed_gcs between 3 - 8 and trauma level 2",coma_state_level2_count
print "For coma state ed_gcs between 3 - 8 and trauma level 3", coma_state_level3_count
print "For coma state ed_gcs between 3 - 8 and trauma level N",coma_state_levelN_count


#for moderate state
moderate_state_level1_count = 0
for i in range(0,len(ED_GCS)):
    if int(ED_GCS[i]) >8  and int(ED_GCS[i]) <= 12 and int(trauma_level[i]) == 1:
        moderate_state_level1_count += 1



moderate_state_level2_count = 0
for i in range(0,len(ED_GCS)):
    if int(ED_GCS[i]) > 8 and int(ED_GCS[i]) <= 12 and int(trauma_level[i]) == 2:
        moderate_state_level2_count += 1


moderate_state_level3_count = 0
for i in range(0, len(ED_GCS)):
    if int(ED_GCS[i]) > 8 and int(ED_GCS[i]) <= 12 and int(trauma_level[i]) == 3:
        moderate_state_level3_count += 1

moderate_state_levelN_count = 0
for i in range(0, len(ED_GCS)):
    if int(ED_GCS[i]) > 8 and int(ED_GCS[i]) <= 12 and int(trauma_level[i]) == 4:
        moderate_state_levelN_count += 1


print "For moderate state ed_gcs between 9 - 12 and trauma level 1",moderate_state_level1_count
print "For moderate state ed_gcs between 9 - 12 and trauma level 2",moderate_state_level2_count
print "For moderate state ed_gcs between 9 - 12 and trauma level 3", moderate_state_level3_count
print "For moderate state ed_gcs between 9 - 12 and trauma level N",moderate_state_levelN_count



#stable state
stable_state_level1_count = 0
for i in range(0,len(ED_GCS)):
    if int(ED_GCS[i]) > 12 and int(trauma_level[i]) == 1:
        stable_state_level1_count += 1



stable_state_level2_count = 0
for i in range(0,len(ED_GCS)):
    if int(ED_GCS[i]) > 12 and int(trauma_level[i]) == 2:
        moderate_state_level2_count += 1


stable_state_level3_count = 0
for i in range(0, len(ED_GCS)):
    if int(ED_GCS[i]) > 12 and int(trauma_level[i]) == 3:
        stable_state_level3_count += 1

stable_state_levelN_count = 0
for i in range(0, len(ED_GCS)):
    if int(ED_GCS[i]) > 12 and int(trauma_level[i]) == 4:
        stable_state_levelN_count += 1



print "For moderate state ed_gcs > 12 and trauma level 1",stable_state_level1_count
print "For moderate state ed_gcs > 12 and trauma level 2",stable_state_level2_count
print "For moderate state ed_gcs > 12 and trauma level 3", stable_state_level3_count
print "For moderate state ed_gcs > 12 and trauma level N",stable_state_levelN_count


#death and fall height:

dead_or_alive = df.iloc[:,38]


#alive - height
count_height6_live = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) <= 6 and dead_or_alive[i] == 'L':
        count_height6_live += 1


count_height6to15_live = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 6 and int(fall_height[i]) < 15 and dead_or_alive[i] == 'L':
        count_height6to15_live += 1


count_height_gt_than_15_live = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 15 and dead_or_alive[i] == 'L':
        count_height_gt_than_15_live += 1

print count_height6_live
print count_height6to15_live
print count_height_gt_than_15_live




#dead - height

count_height6_dead = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) <= 6 and dead_or_alive[i] == 'D':
        count_height6_dead += 1


count_height6to15_dead = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 6 and int(fall_height[i]) < 15 and dead_or_alive[i] == 'D':
        count_height6to15_dead += 1


count_height_gt_than_15_dead = 0
for i in range(0,len(fall_height)):
    if int(fall_height[i]) > 15 and dead_or_alive[i] == 'D':
        count_height_gt_than_15_dead += 1

print count_height6_dead
print count_height6to15_dead
print count_height_gt_than_15_dead







