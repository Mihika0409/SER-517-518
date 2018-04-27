import pandas as pd

incoming_df = pd.read_csv("SampleInput.csv")

df_others = incoming_df[incoming_df['ICD-10 E-code'].str.contains(r'^(?:(?!FALL|MOTOR|CAR).)*$', na=False)]
df_fall = incoming_df[incoming_df['ICD-10 E-code'].str.contains('FALL', na=False)]
df_mv = incoming_df[incoming_df['ICD-10 E-code'].str.contains("MOTOR", na=False) |
                    incoming_df['ICD-10 E-code'].str.contains("CRASH", na=False) |
                    incoming_df['ICD-10 E-code'].str.contains("VEHICLE", na=False) |
                    incoming_df['ICD-10 E-code'].str.contains("CYCLE", na=False) |
                    incoming_df['ICD-10 E-code'].str.contains("TRAFFIC", na=False) |
                    incoming_df['ICD-10 E-code'].str.contains("PASSENGER", na=False)]

if not df_fall.empty:
    print "Fall"
    execfile("Fall.py")
elif not df_mv.empty:
    print "Motor Vehicle Accident"
    if incoming_df['Injury Comments'].str.contains('intubat', na=False).any():
        incoming_df['Intubated'] = 1
    else:
        incoming_df['Intubated'] = 0
    execfile("MotorVehicle.py")
else:
    print "Other Injury"
    execfile("Others.py")
