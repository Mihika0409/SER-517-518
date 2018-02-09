import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('capstoneData.csv', header = None, error_bad_lines=False)
df.columns = ['T1',	'ED/Hosp Arrival Date',	'Age in Years',	'Gender',	'Levels',	'ICD-10 E-code',	'Trauma Type',	'Report of physical abuse',	'Injury Comments',	'Airbag Deployment',	'Patient Position in Vehicle',
              'Safet Equipment Issues',	'Child Restraint',	'MV Speed',	'Fall Height',	'Transport Type',	'Transport Mode',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'Resp Assistance',
              'RTS',	'Field GCS',	'Arrived From',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',
              'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',

              'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005']




df = df[pd.notnull(df['T1'])]





#print df.head()

gender_column = df.iloc[:,3].values


le = LabelEncoder ()
gender_column = le.fit_transform(gender_column)


genderlist = []
for i in gender_column:
    genderlist.append(i)


df['Gender_Numerical'] = genderlist


#print df.head()

trauma_type = df.iloc[:,6].values
#print trauma_type
burnlist = []

for i in trauma_type:
    if i == 'Burn':
        burnlist.append(1)
    else:
        burnlist.append(0)



Blunt_List = []
for i in trauma_type:
    if i  == 'Blunt':
        Blunt_List.append(1)
    else:
        Blunt_List.append(0)


Penetration_List = []
for i in trauma_type:
    if i == 'Penetrating':
        Penetration_List.append(1)
    else:
        Penetration_List.append(0)


df['Burn(Yes/No)'] = burnlist
df['Blunt (Yes/No)'] = Blunt_List
df['Penetrating(Yes/No)'] = Penetration_List






ReportPhysicalAbuse = df.iloc[:,7]

ReportPhysicalAbuse = le.fit_transform(ReportPhysicalAbuse)

report_physical_abuse_list = []

for i in ReportPhysicalAbuse:
    report_physical_abuse_list.append(i)

df['Report of Physical Abuse(Yes/No)'] = report_physical_abuse_list

df.drop(['Gender', 'Trauma Type'], axis = 1, inplace=True)
df.drop(['Report of physical abuse'], axis=1, inplace=True)

#print df.head()

airbag_deployment = df.iloc[:,2].values
airbag_deployment_list = []

for i in airbag_deployment:
    if i == '*NA' or i == 'UNSPECIF_DEPLOY':
        airbag_deployment_list.append(0)
    else:
        airbag_deployment_list.append(1)
#print airbag_deployment_list

df['Airbag Deployment (Yes/No)'] = airbag_deployment_list


front_deployment_list = []
for i in airbag_deployment:
    if i == 'FRONT_DEPLOY':
        front_deployment_list.append(1)
    else:
        front_deployment_list.append(0)
df['Front Deployment (Yes/No)'] = front_deployment_list





side_deployment_list = []

for i in airbag_deployment:
    if i == 'SIDE_DEPLOY':
        side_deployment_list.append(1)
    else:
        side_deployment_list.append(0)

df['Side Deployment (Yes/No)'] = side_deployment_list

df.drop(['Airbag Deployment'], axis = 1, inplace=True)



motor_vehicle_speed  = df.iloc[:,9]
motor_vehicle_list = []
for i in motor_vehicle_speed:
    if i =='*NA' or i == '*ND':
        motor_vehicle_list.append(0)
    else:
        motor_vehicle_list.append(i)

df['Motor_Vehicle_Speed'] = motor_vehicle_list

df.drop(['MV Speed'], axis=1 , inplace = True)



fall_height  = df.iloc[:,9]

fall_height_list = []
for i in fall_height:
    if i =='*NA' or i == '*ND':
        fall_height_list.append(0)
    else:
        fall_height_list.append(i)

df['Fall_Height'] = fall_height_list

df.drop(['Fall Height'], axis=1 , inplace = True)



transportation_type = df.iloc[:,10].values
#print transportation_type

ground_ambulance = []

for i in transportation_type:
    if i == 'Ground Ambulance':
        ground_ambulance.append(1)

    else:
        ground_ambulance.append(0)


Helicopter_ambulance = []

for i in transportation_type:
    if i == 'Helicopter Ambulance':
        Helicopter_ambulance.append(1)

    else:
        Helicopter_ambulance.append(0)


WalkIn_List = []
for i in transportation_type:
    if i == 'Private/Public Vehicle/Walk-in':
        WalkIn_List.append(1)

    else:
        WalkIn_List.append(0)



df['Ground Ambulance (Yes/No)'] = ground_ambulance
df['Helicopter Ambulance (Yes/No)'] = Helicopter_ambulance
df['Walk-in/Pvt/Public (Yes/No)'] = WalkIn_List

df.drop(['Transport Mode'], axis = 1, inplace = True)



respiratory_assistance = df.iloc[:,14].values

assisted_Resp_Rate_List = []
for i in respiratory_assistance:
    if i == 'ssisted Respiratory Rate':
        assisted_Resp_Rate_List.append(1)
    else:
        assisted_Resp_Rate_List.append(0)

df['Resp Assistance (Yes/No)']  = assisted_Resp_Rate_List

df.drop(['Resp Assistance'], axis = 1, inplace = True)

#print df.head()


patient_position = df.iloc[:,6].values


Front_Seat_Passenger_List = []

for i in patient_position:
    if i == 'Front Seat Passenger':
        Front_Seat_Passenger_List.append(1)
    else:
        Front_Seat_Passenger_List.append(0)

df['Patient at Front Seat (Yes/No)'] = Front_Seat_Passenger_List


back_seat = []
for i in patient_position:
    if i == 'Back Seat Passenger (anyone inside except front seat)':
        back_seat.append(1)
    else:
        back_seat.append(0)

df['Patient at Back Seat (Yes/No)'] = back_seat





ATV = []

for i in patient_position:
    if i == 'ATV/Quad (3 or more low press. tires/straddleseat/handlebar)':
        ATV.append(1)
    else:
        ATV.append(0)


df['Patient at ATV/Quad (3 or more low press. tires/straddleseat/handlebar) (Yes/No)'] = ATV




Driver = []

for i in patient_position:
    if i == 'Driver of Motor Vehicle (auto/truck/van) - not motorcycle':
        Driver.append(1)
    else:
        Driver.append(0)

df['Driver of motor vehicle (not bike)'] = Driver


Bicyclist = []

for i in patient_position:
    if i =='Bicyclist (non-motorized)':

        Bicyclist.append(1)
    else:
        Bicyclist.append(0)


df['Position - Biclyclist'] = Bicyclist



Motorcyclist = []


for i in patient_position:
    if i =='Motorcycle Driver (includes mopeds and scooters)':
        Motorcyclist.append(1)
    else:
        Motorcyclist.append(0)


df['Position - Motorcyclist'] = Motorcyclist

# Back of Pickup Truck

dirt_bike = []
for i in patient_position:
    if i =='Dirt Bike, Trail Motorcycle (2 wheel, designed for off-road)':
        dirt_bike.append(1)

    else:
        dirt_bike.append(0)
df[' Position -Dirt Bike, Trail Motorcycle (2 wheel, designed for off-road)'] = dirt_bike


rhino_utv = []
for i in patient_position:
    if i == 'Rhino, Side by Side, UTV (steering wheel/non-straddle seats)':
        rhino_utv.append(1)

    else:
        rhino_utv.append(0)
df['Position - Rhino, Side by Side, UTV (steering wheel/non-straddle seats)'] = rhino_utv



go_kart = []

for i in patient_position:
    if i == 'Go-Kart':
        go_kart.append(1)
    else:
        go_kart.append(0)

df['Positon - Go-Kart'] = go_kart

df.drop(['Patient Position in Vehicle'], axis = 1, inplace = True)

#print df.head()



safety_equipment_issues = df.iloc[:,6].values
safety_equipment = []

count = 0
for i in safety_equipment_issues:
    if i == '*NA':
        safety_equipment.append(0)
    else:
        safety_equipment.append(1)

df['Safety Equipment Issues (Yes/No)'] = safety_equipment

df.drop(['Safet Equipment Issues'], axis = 1, inplace=True)


arrived_from = df.iloc[:,14].values


Referring_Hospital = []



for i in arrived_from:
    if i =='Referring Hospital':
        Referring_Hospital.append(1)
    else:
        Referring_Hospital.append(0)

df['Arrived from Referring Hospital (Yes/No)'] = Referring_Hospital


Scene_of_injury = []


for i in arrived_from:
    if i == 'Scene of Injury':
        Scene_of_injury.append(1)
    else:
        Scene_of_injury.append(0)

df['Arrived from Scene of Injury'] = Scene_of_injury



Home = []

for i in arrived_from:
    if i == 'Home':
        Home.append(1)

    else:
        Home.append(0)

df['Arrived from Home'] = Home




Urgent_Care = []
for i in arrived_from:
    if i == 'Urgent Care':
        Urgent_Care.append(1)
    else:
        Urgent_Care.append(0)

df['Urgent Care'] = Urgent_Care



Clinic = []
for i in arrived_from:
    if i == 'Clinic/MD Office':
        Clinic.append(1)
    else:
        Clinic.append(0)

df['Arrived from Clinic'] = Clinic


df.drop(['Arrived From'], axis=1, inplace=True)

#df.drop(['T1','ED/Hosp Arrival Date','Dispositon from  ED','Received blood within 4 hrs','Severe Brain Injury',	'Time to 1st OR Visit (mins.)','Discharge Disposition'], axis=1, inplace = True)




#print X.head()


glasgow = df.iloc[:,13].values
print glasgow
Severe_Head_Injury  = []


for i in glasgow:
    if i < 9:
        Severe_Head_Injury.append(1)
    else:
        Severe_Head_Injury.append(0)

df['Severe Head Injury - GCS < 9'] = Severe_Head_Injury

df.drop(['Field GCS'], axis=1, inplace=True)

X = df.loc[df['Levels'].isin(['1','2'])]

#print X.head()



writer = pd.ExcelWriter('output.xlsx')
X.to_excel(writer,'Sheet1')
X.to_excel(writer,'Sheet2')
writer.save()


























