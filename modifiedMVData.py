import pandas as pd
import csv


#Code to check if a number if float or not.
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#Code to catogerize motor vehicle data.
with open('/Users/satishnandan/Desktop/TraumaActivation/newData1.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/m2.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if ('accident' in row[6].strip().lower() and 'traffic' in row[6].strip().lower()) or ('accident' in row[5].strip().lower() and 'traffic' in row[5].strip().lower()):
            if (row[4] == '1' or row[4] == '2'):
                if row[17].strip() != 'Private/Public Vehicle/Walk-in':
                    # if row[18].isdigit() and row[19].isdigit() and row[20].isdigit() and row[23].isdigit() and isfloat(row[22]):
                    if row[4].strip() == "2":
                        row[4] = "0"
                    if row[3].strip() == "M":
                        row[3] = "1"
                    elif row[3].strip() == "F":
                        row[3] = "0"
                    writer.writerow(row)
#Code to find the mean of columns.
SBP = 0
HR = 0
RR = 0
RTS = 0
GCS = 0
with open('/Users/satishnandan/Desktop/TraumaActivation/mvspeed10.csv', 'rb') as inp:
    for row in csv.reader(inp):
        if isfloat(row[18]):
            SBP = SBP + float(row[18])
        if isfloat(row[19]):
            HR = HR + float(row[19])
        if isfloat(row[20]):
            RR = RR +float(row[20])
        if isfloat(row[22]):
            RTS = RTS + float(row[22])
        if isfloat(row[23]):
            GCS = GCS + float(row[23])
print SBP/1048
print HR/1048
print RR/1048
print RTS/1048
print GCS/1048

#Code to append a new intubated column to data.
with open('/Users/satishnandan/Desktop/TraumaActivation/m2.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/m1.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if "intubated" in row[9].lower():
            row.append('1')
        else:
            row.append('0')
        writer.writerow(row)
 
#Modified the RespAssistance Column.
with open('/Users/satishnandan/Desktop/TraumaActivation/m1.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/m4.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
      if row[21] == "Assisted Respiratory Rate":
          row[21] = '1'
      else:
          row[21] = '0'
      writer.writerow(row)
        
#Data frame to replace blank values with mean values.        
fields = ['Age in Years', 'Gender', 'MV Speed','RTS','Field GCS','Field SBP', 'Field HR', 'Field RR','Intubated','Resp Assistance']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/m1.csv',
    usecols=fields)
fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/m1.csv',
    usecols=fields1)
df['RTS'] = df['RTS'].replace(['*NA','*ND','*BL',''],'7.65')
df['Field GCS']=df['Field GCS'].replace(['*NA','*ND','*BL',''],'14.54')
df['Field SBP']=df['Field SBP'].replace(['*NA','*ND','*BL',''],'119')
df['Field HR']=df['Field HR'].replace(['*NA','*ND','*BL',''],'110')
df['Field RR']=df['Field RR'].replace(['*NA','*ND','*BL',''],'21')
df['MV Speed'] = df['MV Speed'].replace(['*NA','*ND','*BL',''],'0')
