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
# # Finding out the patterns for traumaType = Blunt

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

