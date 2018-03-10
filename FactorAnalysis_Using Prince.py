import pandas as pd
from sklearn.decomposition import FactorAnalysis

from sklearn import decomposition, preprocessing

import numpy as np

import matplotlib.pyplot as plot

import pandas as pd

import prince

df = pd.read_csv('newoutputfile.csv', header = None, error_bad_lines=False)

df.columns =['PatientID','T1','ED/Hosp Arrival Dategg',	'Age in Yearss',	'Levels',	'ICD-10 E-code',	'Injury Comments',	'Child Restraint'	,'Transport Type',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'RTS',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',	'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',	'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005',	'Gender_Numerical',	'Burn(Yes/No)',	'Blunt (Yes/No)',	'Penetrating(Yes/No)',	'Report of Physical Abuse(Yes/No)',	'Airbag Deployment (Yes/No)',	'Front Deployment (Yes/No)',	'Side Deployment (Yes/No)',	'Motor_Vehicle_Speed',	'Fall_Height',	'Ground Ambulance (Yes/No)',	'Helicopter Ambulance (Yes/No)',	'Walk-in/Pvt/Public (Yes/No)',	'Resp Assistance (Yes/No)',	'Patient at Front Seat (Yes/No)',	'Patient at Back Seat (Yes/No)',	'Patient at ATV/Quad (3 or more low press. tires/straddleseat/handlebar) (Yes/No)',	'Driver of motor vehicle (not bike)',	'Position - Biclyclist',	'Position - Motorcyclist',	 'Position -Dirt Bike Trail Motorcycle (2 wheel, designed for off-road)',	'Position - Rhino, Side by Side, UTV (steering wheel/non-straddle seats)',	'Positon - Go-Kart',	'Safety Equipment Issues (Yes/No)',	'Arrived from Referring Hospital (Yes/No)',	'Arrived from Scene of Injury',	'Arrived from Home',	'Urgent Care',	'Arrived from Clinic',	'Severe Head Injury - GCS < 9']

X = df[['Total Vent Days',	'Total Days in ICU','Admission Hosp LOS (days)','Severe Brain Injury']]

pca = prince.CA(X, n_components=4)

fig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='class', ellipse_fill=True)

plot.show()

ig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='class', ellipse_fill=True)
fig3, ax3 = pca.plot_correlation_circle()

fig1.savefig('pca_cumulative_inertia.png', bbox_inches='tight', pad_inches=1)
fig2.savefig('pca_row_principal_coordinates.png', bbox_inches='tight', pad_inches=1)
fig3.savefig('pca_correlation_circle.png', bbox_inches='tight', pad_inches=1)
