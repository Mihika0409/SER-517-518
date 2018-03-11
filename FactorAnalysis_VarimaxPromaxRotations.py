import pandas as pd

from factor_analyzer import FactorAnalyzer

df = pd.read_csv('newoutputfile.csv', header = None, error_bad_lines=False)

df.columns =['PatientID','T1','ED/Hosp Arrival Dategg',	'Age in Yearss',	'Levels',	'ICD-10 E-code',	'Injury Comments',	'Child Restraint'	,'Transport Type',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'RTS',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',	'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',	'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005',	'Gender_Numerical',	'Burn(Yes/No)',	'Blunt (Yes/No)',	'Penetrating(Yes/No)',	'Report of Physical Abuse(Yes/No)',	'Airbag Deployment (Yes/No)',	'Front Deployment (Yes/No)',	'Side Deployment (Yes/No)',	'Motor_Vehicle_Speed',	'Fall_Height',	'Ground Ambulance (Yes/No)',	'Helicopter Ambulance (Yes/No)',	'Walk-in/Pvt/Public (Yes/No)',	'Resp Assistance (Yes/No)',	'Patient at Front Seat (Yes/No)',	'Patient at Back Seat (Yes/No)',	'Patient at ATV/Quad (3 or more low press. tires/straddleseat/handlebar) (Yes/No)',	'Driver of motor vehicle (not bike)',	'Position - Biclyclist',	'Position - Motorcyclist',	 'Position -Dirt Bike Trail Motorcycle (2 wheel, designed for off-road)',	'Position - Rhino, Side by Side, UTV (steering wheel/non-straddle seats)',	'Positon - Go-Kart',	'Safety Equipment Issues (Yes/No)',	'Arrived from Referring Hospital (Yes/No)',	'Arrived from Scene of Injury',	'Arrived from Home',	'Urgent Care',	'Arrived from Clinic',	'Severe Head Injury - GCS < 9']


factoranalyser = FactorAnalyzer()

X = df[['Total Vent Days',	'Total Days in ICU','Admission Hosp LOS (days)']]


#Factor Analysis and implementing Varimax Rotation

print("Results after varimax rotation")

factoranalyser.analyze(X,3, rotation= 'varimax')

print (factoranalyser.loadings)

print (factoranalyser.get_uniqueness())

print (factoranalyser.get_factor_variance())

print (factoranalyser.get_eigenvalues())



#Factor Analysis and Implementing Promax Rotation

factoranalyser.analyze(X,3, rotation= 'promax')

print("Results after promax rotation")

print (factoranalyser.loadings)

print (factoranalyser.get_uniqueness())

print (factoranalyser.get_factor_variance())

print (factoranalyser.get_eigenvalues())


#Factor Analysis without rotating the matrix

print("Results without rotating the matrix")

factoranalyser.analyze(X,3, rotation= None)

print (factoranalyser.loadings)

print (factoranalyser.get_uniqueness())

print (factoranalyser.get_factor_variance())

print (factoranalyser.get_eigenvalues())

