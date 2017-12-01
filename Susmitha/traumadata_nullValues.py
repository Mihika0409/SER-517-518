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
total = na_count_fallheight + nd_count_fallheight + bl_count_fallheight
print ('--------------------------')
fall_height_nullvalues_percent = (total/float(len(fall_height)))*100
print (fall_height_nullvalues_percent)

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
total = na_count_traumatype + nd_count_traumatype + bl_count_traumatype
print ('--------------------------')
trauma_type_nullvalues_percent = (total/float(len(trauma_type)))*100
print (trauma_type_nullvalues_percent)

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
total = na_count_airbagdeploy + nd_count_airbagdeploy + bl_count_airbagdeploy
print ('--------------------------')
airbag_deploy_nullvalues_percent = (total/float(len(airbag_deploy)))*100
print (airbag_deploy_nullvalues_percent)

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
total = na_count_patientpos + nd_count_patientpos + bl_count_patientpos
print ('--------------------------')
patient_pos_nullvalues_percent = (total/float(len(patient_pos)))*100
print (patient_pos_nullvalues_percent)

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
total = na_count_safety_equip_issues + nd_count_safety_equip_issues + bl_count_safety_equip_issues
print ('--------------------------')
safety_equip_issues_nullvalues_percent = (total/float(len(safety_equip_issues)))*100
print (safety_equip_issues_nullvalues_percent)

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
total = na_count_child_restraint + nd_count_child_restraint + bl_count_child_restraint
print ('--------------------------')
child_restraint_nullvalues_percent = (total/float(len(child_restraint)))*100
print (child_restraint_nullvalues_percent)

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
total = na_count_mv_speed + nd_count_mv_speed + bl_count_mv_speed
print ('--------------------------')
mv_speed_nullvalues_percent = (total/float(len(mv_speed)))*100
print (mv_speed_nullvalues_percent)

transport_mode = df['transport_mode']
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
print (bl_count_transport_mode)
total = na_count_transport_mode + nd_count_transport_mode + bl_count_transport_mode
print ('--------------------------')
transport_mode_nullvalues_percent = (total/float(len(transport_mode)))*100
print (transport_mode_nullvalues_percent)

field_SBP1 = df['field_SBP']
print (field_SBP1)
na_count_field_SBP = 0
nd_count_field_SBP = 0
bl_count_field_SBP = 0
for i in range(0, len(field_SBP1)):
    if field_SBP1[i] == "*NA":
        na_count_field_SBP = na_count_field_SBP + 1

    elif field_SBP1[i] == "*ND":
        nd_count_field_SBP = nd_count_field_SBP + 1

    elif field_SBP1[i] == "*BL":
        bl_count_field_SBP = bl_count_field_SBP + 1
print (na_count_field_SBP)
total = na_count_field_SBP + nd_count_field_SBP + bl_count_field_SBP
print ('--------------------------')
field_SBP1_nullvalues_percent = (total/float(len(field_SBP1)))*100
print (field_SBP1_nullvalues_percent)

total_vent_days = df['total_vent_days']
print (total_vent_days)
na_count_total_vent_days = 0
nd_count_total_vent_days = 0
bl_count_total_vent_days = 0
for i in range(0, len(total_vent_days)):
    if total_vent_days[i] == "*NA":
        na_count_total_vent_days = na_count_total_vent_days + 1

    elif total_vent_days[i] == "*ND":
        nd_count_total_vent_days = nd_count_total_vent_days + 1

    elif total_vent_days[i] == "*BL":
        bl_count_total_vent_days = bl_count_total_vent_days + 1
print (na_count_total_vent_days)
total = na_count_total_vent_days + nd_count_total_vent_days + bl_count_total_vent_days
print ('--------------------------')
total_vent_days_nullvalues_percent = (total/float(len(total_vent_days)))*100
print (total_vent_days_nullvalues_percent)

field_HR = df['field_HR']
print (field_HR)
na_count_field_HR = 0
nd_count_field_HR = 0
bl_count_field_HR = 0
for i in range(0, len(field_HR)):
    if field_HR[i] == "*NA":
        na_count_field_HR = na_count_field_HR + 1

    elif field_HR[i] == "*ND":
        nd_count_field_HR = nd_count_field_HR + 1

    elif field_HR[i] == "*BL":
        bl_count_field_HR = bl_count_field_HR + 1
print (na_count_field_HR)
total = na_count_field_HR + nd_count_field_HR + bl_count_field_HR
print ('--------------------------')
field_HR_nullvalues_percent = (total/float(len(field_HR)))*100
print (field_HR_nullvalues_percent)

field_shock_ind = df['field_shock_ind']
print (field_shock_ind)
na_count_field_shock_ind = 0
nd_count_field_shock_ind = 0
bl_count_field_shock_ind = 0
for i in range(0, len(field_shock_ind)):
    if field_shock_ind[i] == "*NA":
        na_count_field_shock_ind = na_count_field_shock_ind + 1

    elif field_shock_ind[i] == "*ND":
        nd_count_field_shock_ind = nd_count_field_shock_ind + 1

    elif field_shock_ind[i] == "*BL":
        bl_count_field_shock_ind = bl_count_field_shock_ind + 1
print (na_count_field_shock_ind)
total = na_count_field_shock_ind + nd_count_field_shock_ind + bl_count_field_shock_ind
print ('--------------------------')
field_shock_ind_nullvalues_percent = (total/float(len(field_shock_ind)))*100
print (field_shock_ind_nullvalues_percent)

ED_GCS = df['ED_GCS']
print (ED_GCS)
na_count_ED_GCS = 0
nd_count_ED_GCS = 0
bl_count_ED_GCS = 0
for i in range(0, len(ED_GCS)):
    if ED_GCS[i] == "*NA":
        na_count_ED_GCS = na_count_ED_GCS + 1

    elif ED_GCS[i] == "*ND":
        nd_count_ED_GCS = nd_count_ED_GCS + 1

    elif ED_GCS[i] == "*BL":
        bl_count_ED_GCS = bl_count_ED_GCS + 1
print (na_count_ED_GCS)
total = na_count_ED_GCS + nd_count_ED_GCS + bl_count_ED_GCS
print ('--------------------------')
ED_GCS_nullvalues_percent = (total/float(len(ED_GCS)))*100
print (ED_GCS_nullvalues_percent)

brain_injury = df['brain_injury']
print (brain_injury)
na_count_brain_injury = 0
nd_count_brain_injury = 0
bl_count_brain_injury = 0
for i in range(0, len(brain_injury)):
    if brain_injury[i] == "*NA":
        na_count_brain_injury = na_count_brain_injury + 1

    elif brain_injury[i] == "*ND":
        nd_count_brain_injury = nd_count_brain_injury + 1

    elif brain_injury[i] == "*BL":
        bl_count_brain_injury = bl_count_brain_injury + 1
print (na_count_brain_injury)
total = na_count_brain_injury + nd_count_brain_injury + bl_count_brain_injury
print ('--------------------------')
brain_injury_nullvalues_percent = (total/float(len(brain_injury)))*100
print (brain_injury_nullvalues_percent)
