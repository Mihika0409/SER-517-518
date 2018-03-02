import pandas as pd
from sklearn.decomposition import FactorAnalysis

from sklearn import decomposition, preprocessing

import numpy as np

import pandas as pd

df = pd.read_csv('newoutputfile.csv', header = None, error_bad_lines=False)

df.columns =['PatientID','T1','ED/Hosp Arrival Dategg',	'Age in Yearss',	'Levels',	'ICD-10 E-code',	'Injury Comments',	'Child Restraint'	,'Transport Type',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'RTS',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',	'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',	'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005',	'Gender_Numerical',	'Burn(Yes/No)',	'Blunt (Yes/No)',	'Penetrating(Yes/No)',	'Report of Physical Abuse(Yes/No)',	'Airbag Deployment (Yes/No)',	'Front Deployment (Yes/No)',	'Side Deployment (Yes/No)',	'Motor_Vehicle_Speed',	'Fall_Height',	'Ground Ambulance (Yes/No)',	'Helicopter Ambulance (Yes/No)',	'Walk-in/Pvt/Public (Yes/No)',	'Resp Assistance (Yes/No)',	'Patient at Front Seat (Yes/No)',	'Patient at Back Seat (Yes/No)',	'Patient at ATV/Quad (3 or more low press. tires/straddleseat/handlebar) (Yes/No)',	'Driver of motor vehicle (not bike)',	'Position - Biclyclist',	'Position - Motorcyclist',	 'Position -Dirt Bike Trail Motorcycle (2 wheel, designed for off-road)',	'Position - Rhino, Side by Side, UTV (steering wheel/non-straddle seats)',	'Positon - Go-Kart',	'Safety Equipment Issues (Yes/No)',	'Arrived from Referring Hospital (Yes/No)',	'Arrived from Scene of Injury',	'Arrived from Home',	'Urgent Care',	'Arrived from Clinic',	'Severe Head Injury - GCS < 9']

X = df[['Gender_Numerical','Burn(Yes/No)','Blunt (Yes/No)','Penetrating(Yes/No)','Report of Physical Abuse(Yes/No)','Airbag Deployment (Yes/No)','Front Deployment (Yes/No)','Side Deployment (Yes/No)','Motor_Vehicle_Speed','Fall_Height','Ground Ambulance (Yes/No)','Helicopter Ambulance (Yes/No)','Walk-in/Pvt/Public (Yes/No)','Resp Assistance (Yes/No)','Patient at Front Seat (Yes/No)','Patient at Back Seat (Yes/No)','Patient at ATV/Quad (3 or more low press. tires/straddleseat/handlebar) (Yes/No)','Severe Head Injury - GCS < 9']]

variable_names = ['Gender_Numerical','Burn(Yes/No)','Blunt (Yes/No)','Penetrating(Yes/No)','Report of Physical Abuse(Yes/No)','Airbag Deployment (Yes/No)','Front Deployment (Yes/No)','Side Deployment (Yes/No)','Motor_Vehicle_Speed','Fall_Height','Ground Ambulance (Yes/No)','Helicopter Ambulance (Yes/No)','Walk-in/Pvt/Public (Yes/No)','Resp Assistance (Yes/No)','Patient at Front Seat (Yes/No)','Patient at Back Seat (Yes/No)','Patient at ATV/Quad (3 or more low press. tires/straddleseat/handlebar) (Yes/No)','Severe Head Injury - GCS < 9']

#print variable_names

#print X.head(2)
list = []

list = X.values.tolist()

factor = FactorAnalysis().fit(list)

#print pd.DataFrame(factor.components_, columns=variable_names)


X = X[~np.isnan(X).any(axis=1)]

data_normal = preprocessing.scale(X)

fa = decomposition.FactorAnalysis(n_components = 1)

fa.fit(data_normal)

for score in fa.score_samples(data_normal):
    print score






