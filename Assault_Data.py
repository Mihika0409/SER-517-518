import pandas as pd

df = pd.read_csv('assault1.csv', header = None, error_bad_lines=False)
df.columns = ['tid', 'hosp_date', 'age', 'gender', 'levels' , 'icd_code', 'trauma_type' , 'physical_abuse', 'injury_comments', 'airbag_deploy',
              'patient_pos' , 'safety_equip_issues' , 'child_restraint' , 'mv_speed' , 'fall_height', 'transport_type' , 'transport_mode',
              'field_SBP' , 'field_HR' , 'field_shock_ind' , 'field_RR' , 'resp_assis ' , 'RTS' ,'field_GCS' , 'arrived_from' , 'ED_LOS' ,
              'disposition', 'ED_SBP' , 'ED_HR' , 'ED_RR' , 'ED_GCS' , 'total_vent_days', 'days_in_icu', 'hosp_LOS' , 'total_LOS' ,
              'received_blood' , 'brain_injury', 'time_to_first_OR', 'death ', 'discharge_dispo' , 'injury_score' ,'AIS', 'AIS_2005']

#[5 rows x 43 columns]
#print df.head()

traumaType = df.iloc[:, 6]
traumaLevel = df.iloc[:, 4]

print traumaType
print "-----------------------"
print traumaLevel

pen_count = 0;
pen_count_level1 = 0;
pen_count_level2 = 0;
pen_count_level3 = 0;
pen_count_levelN = 0;


# Finding out the patterns for traumaType = Penetration
for i in range(0, len(traumaType)):
    if traumaType[i] == 'Penetratin':
        pen_count = pen_count + 1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Penetratin' and traumaLevel[i] == '1':
        pen_count_level1 = pen_count_level1 +1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Penetratin' and traumaLevel[i] == '2':
        pen_count_level2 = pen_count_level2 +1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Penetratin' and traumaLevel[i] == '3':
        pen_count_level3 = pen_count_level3 +1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Penetratin' and traumaLevel[i] == 'N':
        pen_count_levelN = pen_count_levelN +1

print "the total count for penetration is: "
print pen_count

print "the level 1 count for penetration is: "
print pen_count_level1

print "the level 2 count for penetration is: "
print pen_count_level2

print "the level 3 count for penetration is: "
print pen_count_level3

print "the level N count for penetration is: "
print pen_count_levelN

print "--------------------------------------------------------------"
# Finding out the patterns for traumaType = Blunt

blunt_count = 0
blunt_count_level1 = 0
blunt_count_level2 = 0
blunt_count_level3 = 0
blunt_count_levelN = 0

for i in range(0, len(traumaType)):
    if traumaType[i] == 'Blunt':
        blunt_count = blunt_count + 1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Blunt' and traumaLevel[i] == '1':
        blunt_count_level1 = blunt_count_level1 +1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Blunt' and traumaLevel[i] == '2':
        blunt_count_level2 = blunt_count_level2 +1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Blunt' and traumaLevel[i] == '3':
        blunt_count_level3 = blunt_count_level3 +1

for i in range(0, len(traumaLevel)):
    if traumaType[i] == 'Blunt' and traumaLevel[i] == 'N':
        blunt_count_levelN = blunt_count_levelN +1

print "the total count for blunt is: "
print blunt_count

print "the level 1 count of blunt is: "
print blunt_count_level1

print "the level 2 count of blunt is: "
print blunt_count_level2

print "the level 3 count of blunt is: "
print blunt_count_level3

print "the level N count of blunt is: "
print blunt_count_levelN

# Finding out patterns for the GCS scale

GCS = df['field_GCS']

gcs_0_2 = 0;
gcs_3_8 = 0;
gcs_9_12 = 0;
gcs_13_15 = 0;

for i in range(0, len(GCS)):
    if int(GCS[i]) >= 0 and int(GCS[i]) <= 2:
        gcs_0_2 = gcs_0_2 + 1

for i in range(0, len(GCS)):
    if int(GCS[i]) >= 3 and int(GCS[i]) <= 8:
        gcs_3_8 = gcs_3_8 + 1

for i in range(0, len(GCS)):
    if int(GCS[i]) >= 9 and int(GCS[i]) <= 12:
        gcs_9_12 = gcs_9_12 + 1

for i in range(0, len(GCS)):
    if int(GCS[i]) >= 13 and int(GCS[i]) <= 15:
        gcs_13_15 = gcs_13_15 + 1

print "The number of people suffereing from GCS values from 0-2 are:"
print gcs_0_2

print "The number of people suffereing from GCS values from 3-8 are:"
print gcs_3_8

print "The number of people suffereing from GCS values from 9-12 are:"
print gcs_9_12

print "The number of people suffereing from GCS values from 13-15 are:"
print gcs_13_15

# Visualizing the assault data

import matplotlib.pyplot as plt
import numpy as np

objects = ("Penetrating", "Blunt")
y_pos = np.arange(len(objects))
counts = [pen_count, blunt_count]
plt.bar(y_pos, counts, align='center', alpha=0.5, width=0.35)
plt.xticks(y_pos, objects)
plt.ylabel('Trauma')
plt.show()

# Visualizing the GCS data

objects1 = ("Level: 0-2", "Level: 3-8", "Level: 9-12", "Level: 13-15")
y_pos = np.arange(len(objects1))
counts1 = [gcs_0_2, gcs_3_8, gcs_9_12, gcs_13_15]
plt.bar(y_pos, counts1, align='center', alpha=0.5, width=0.35)
plt.xticks(y_pos, objects1)
plt.ylabel('Count')
plt.show()

# Plots to show blunt and penetration data side by side
# data to plot
n_groups = 4
means_penetration = (pen_count_level1, pen_count_level2, pen_count_level3, pen_count_levelN)
means_blunt = (blunt_count_level1, blunt_count_level2, blunt_count_level3, blunt_count_levelN)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_penetration, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Penetration')

rects2 = plt.bar(index + bar_width, means_blunt, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Blunt')

plt.xlabel('Types of Assault')
plt.ylabel('Scores')
plt.title('Assault Data')
plt.xticks(index + bar_width, ('Level1', 'Level2', 'Level3', 'LevelN'))
plt.legend()

plt.tight_layout()
plt.show()

# Plot to show the line graph for field SBP

print '-----------------'
sbp = df['field_SBP']
level1_sbp = []
level2_sbp = []
level3_sbp = []
levelN_sbp = []


for i in range(0, len(sbp)):
    if traumaLevel[i] == '1':
        level1_sbp.append(sbp[i])
    elif traumaLevel[i] == '2':
        level2_sbp.append(sbp[i])
    elif traumaLevel[i] == '3':
        level3_sbp.append(sbp[i])
    elif traumaLevel[i] == 'N':
        levelN_sbp.append(sbp[i])

from pylab import *

t1 = arange(len(level1_sbp))
t2 = arange(len(level2_sbp))
t3 = arange(len(level3_sbp))
tn = arange(len(levelN_sbp))

plot(t1, level1_sbp, color='red', label='level1')
plot(t2, level2_sbp, color='blue', label='level2')
plot(t3, level3_sbp, color='yellow', label='level3')
plot(tn, levelN_sbp, color='black', label='levelN')

xlabel('Item (s)')
ylabel('SBP Value')
title('Trauma Levels for Assault vs the Field SBP Levels')
grid(True)
legend()
show()

# Plot to show the line graph for Total Length of stay in the hospital

LOS = df['total_LOS']
level1_los = []
level2_los = []
level3_los = []
levelN_los = []

for i in range(0, len(LOS)):
    if traumaLevel[i] == '1':
        level1_los.append(LOS[i])
    elif traumaLevel[i] == '2':
        level2_los.append(LOS[i])
    elif traumaLevel[i] == '3':
        level3_los.append(LOS[i])
    elif traumaLevel[i] == 'N':
        levelN_los.append(LOS[i])

l1 = arange(len(level1_los))
l2 = arange(len(level2_los))
l3 = arange(len(level3_los))
ln = arange(len(levelN_los))

plot(l1, level1_los, color='red', label='level1')
plot(l2, level2_los, color='blue', label='level2')
plot(l3, level3_los, color='yellow', label='level3')
plot(ln, levelN_los, color='black', label='levelN')

xlabel('Item (s)')
ylabel('Length of Stay (in days)')
title('Trauma Levels for Assault vs Length of stay in the hospital')
grid(True)
legend()
show()
