import pandas as pd

# df = pd.read_csv("CleanData.csv")
# df = df[['T1#', 'Injury Comments']]

'''df.columns = ['T1', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code', 'ICD-9 E-code',
              'Trauma Type', 'Report of physical abuse', 'Injury Comments', 'Airbag Deployment',
              'Patient Position in Vehicle', 'Safet Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height',
              'Transport Type', 'Transport Mode', 'Field SBP', 'Field HR', 'Field RR', 'Resp Assistance', 'RTS',
              'Field GCS', 'Arrived From', 'ED LOS (mins)', 'Dispositon from  ED', 'ED SBP', 'ED HR', 'ED RR', 'ED GCS',
              'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
              'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive', 'Discharge Disposition', 'Injury Severity Score', 'ICD-9', 'ICD-10',
              'AIS 2005']'''

fields = ['Age in Years', 'Gender', 'ICD-10 E-code', 'Trauma Type', 'Injury Comments', 'MV Speed', 'Fall Height',
          'Field SBP', 'Field HR', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS']

incoming_patient = {'Age in Years': 6, 'Gender': 'M', 'ICD-10 E-code': 'BITTEN BY DOG, INITIAL ENCOUNTER',
                    'Trauma Type': 'Penetrating',
                    'Injury Comments': 'presents from UC after dog bite to R eye. At about 1045 this AM, patient was bit by family pet dog (husky). No LOC. Pt sustained laceration to R eye and was evaluated at urgent care. There, provider was unable to assess pupil but noted extensive lacerations so gave pt hydrocodone and transferred here for further management. Pt denies HA, nausea/vomiting, and states that he has blurred vision in R eye.',
                    'MV Speed': '*NA', 'Fall Height': '*NA', 'Field SBP': '*NA', 'Field HR': '*NA', 'Field RR': '*NA',
                    'Resp Assistance': 'Unassisted Respiratory Rate', 'RTS': '*NA', 'Field GCS': '*NA'}

if ('FALL' in incoming_patient.get('ICD-10 E-code')) or ('FALL' in incoming_patient.get('Injury Comments')):
    print "Fall"
elif ('VEHICLE' in incoming_patient.get('ICD-10 E-code')) or ('VEHICLE' in incoming_patient.get('Injury Comments')):
    print "Motor Vehicle Accident"
else:
    print "Other"
