import pandas as pd

df = pd.read_csv('Trauma2.csv', header = None, error_bad_lines=False)
df.columns = ['tid', 'hosp_date', 'age', 'gender', 'levels' , 'icd_code', 'trauma_type' , 'physical_abuse', 'injury_comments', 'airbag_deploy',
              'patient_pos' , 'safety_equip_issues' , 'child_restraint' , 'mv_speed' , 'fall_height', 'transport_type' , 'transport_mode',
              'field_SBP' , 'field_HR' , 'field_shock_ind' , 'field_RR' , 'resp_assis ' , 'RTS' ,'field_GCS' , 'arrived_from' , 'ED_LOS' ,
              'disposition', 'ED_SBP' , 'ED_HR' , 'ED_RR' , 'ED_GCS' , 'total_vent_days', 'days_in_icu', 'hosp_LOS' , 'total_LOS' ,
              'received_blood' , 'brain_injury', 'time_to_first_OR', 'death ', 'discharge_dispo' , 'injury_score' ,'AIS']

#print(df.head())

fall_height = df['fall_height']
print (fall_height)
na_count_fallheight = 0
nd_count_fallheight = 0
bl_count_fallheight = 0
for i in range(0, len(fall_height)):
    if fall_height[i] == "*NA":
        na_count_fallheight = na_count_fallheight + 1

    elif fall_height[i] == "*ND":
        nd_count_fallheight = nd_count_fallheight + 1

    elif fall_height[i] == "*BL":
        bl_count_fallheight = bl_count_fallheight + 1
print (na_count_fallheight)

trauma_type = df['trauma_type']
print(trauma_type)
na_count_traumatype = 0
nd_count_traumatype = 0
bl_count_traumatype = 0
for i in range(0, len(trauma_type)):
    if trauma_type[i] == "*NA":
        na_count_traumatype = na_count_traumatype + 1

    elif trauma_type[i] == "*ND":
        nd_count_traumatype = nd_count_traumatype + 1

    elif trauma_type[i] == "*BL":
        bl_count_traumatype = bl_count_traumatype + 1
print (nd_count_traumatype)

airbag_deploy = df['airbag_deploy']
print (airbag_deploy)
na_count_airbagdeploy = 0
nd_count_airbagdeploy = 0
bl_count_airbagdeploy = 0
for i in range(0, len(airbag_deploy)):
    if airbag_deploy[i] == "*NA":
        na_count_airbagdeploy = na_count_airbagdeploy + 1

    elif fall_height[i] == "*ND":
        nd_count_airbagdeploy = nd_count_airbagdeploy + 1

    elif fall_height[i] == "*BL":
        bl_count_airbagdeploy = bl_count_airbagdeploy + 1
print (na_count_airbagdeploy)

patient_pos = df['patient_pos']
print (patient_pos)
na_count_patientpos = 0
nd_count_patientpos = 0
bl_count_patientpos = 0
for i in range(0, len(patient_pos)):
    if patient_pos[i] == "*NA":
        na_count_patientpos = na_count_patientpos + 1

    elif patient_pos[i] == "*ND":
        nd_count_patientpos = nd_count_patientpos + 1

    elif patient_pos[i] == "*BL":
        bl_count_patientpos = bl_count_patientpos + 1
print (na_count_patientpos)

safety_equip_issues = df['safety_equip_issues']
print (safety_equip_issues)
na_count_safety_equip_issues = 0
nd_count_safety_equip_issues = 0
bl_count_safety_equip_issues = 0
for i in range(0, len(safety_equip_issues)):
    if safety_equip_issues[i] == "*NA":
        na_count_safety_equip_issues = na_count_safety_equip_issues + 1

    elif safety_equip_issues[i] == "*ND":
        nd_count_safety_equip_issues = nd_count_safety_equip_issues + 1

    elif safety_equip_issues[i] == "*BL":
        bl_count_safety_equip_issues = bl_count_safety_equip_issues + 1
print (na_count_safety_equip_issues)

child_restraint = df['child_restraint']
print (child_restraint)
na_count_child_restraint = 0
nd_count_child_restraint = 0
bl_count_child_restraint = 0
for i in range(0, len(child_restraint)):
    if child_restraint[i] == "*NA":
        na_count_child_restraint = na_count_child_restraint + 1

    elif child_restraint[i] == "*ND":
        nd_count_child_restraint = nd_count_child_restraint + 1

    elif child_restraint[i] == "*BL":
        bl_count_child_restraint = bl_count_child_restraint + 1
print (na_count_child_restraint)

mv_speed = df['mv_speed']
print (mv_speed)
na_count_mv_speed = 0
nd_count_mv_speed = 0
bl_count_mv_speed = 0
for i in range(0, len(mv_speed)):
    if mv_speed[i] == "*NA":
        na_count_mv_speed = na_count_mv_speed + 1

    elif mv_speed[i] == "*ND":
        nd_count_mv_speed = nd_count_mv_speed + 1

    elif mv_speed[i] == "*BL":
        bl_count_mv_speed = bl_count_mv_speed + 1
print (na_count_mv_speed)

'''transport_mode = df['transport_mode']
print (transport_mode)
na_count_transport_mode = 0
nd_count_transport_mode = 0
bl_count_transport_mode = 0
for i in range(0, len(transport_mode)):
    if transport_mode[i] == "*NA":
        na_count_transport_mode = na_count_transport_mode + 1

    elif transport_mode[i] == "*ND":
        nd_count_transport_mode = nd_count_transport_mode + 1

    elif transport_mode[i] == "*BL":
        bl_count_transport_mode = bl_count_transport_mode + 1
print (bl_count_transport_mode)'''