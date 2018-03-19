import csv

#Function to check if a value can be converted into float or not.
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


#To catogerize data into motor vehicle accidents.
with open('/Users/satishnandan/Desktop/TraumaActivation/newData1.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/motor.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if ('accident' in row[6].strip().lower() and 'traffic' in row[6].strip().lower()) or ('accident' in row[5].strip().lower() and 'traffic' in row[5].strip().lower()) or row[14].isdigit():
            if(row[14].isdigit()):
                if(row[4] == '1' or row[4] == '2'):
                    if row[17].strip() != 'Private/Public Vehicle/Walk-in':
                        if row[18].isdigit() and row[19].isdigit() and row[20].isdigit() and row[23].isdigit() and isfloat(row[22]):
                            if row[4].strip() == "2":
                                row[4] = "0"
                            if row[3].strip() == "M":
                                row[3] = "1"
                            elif row[3].strip() == "F":
                                row[3] = "0"
                            writer.writerow(row)
