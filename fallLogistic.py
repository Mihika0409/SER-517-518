import csv
with open('edittedTrauma.csv', 'rb') as inp, open('fall1.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if not 'FALL' in row[5].strip() or row[14].strip() == '*NA' or row[14].strip() == '*ND' or row[14].strip() == '*BL' or row[14].strip() == '':
            continue
        else:
            writer.writerow(row)

with open('fall1.csv', 'rb') as inp, open('fall2.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[16].strip() == 'Private/Public Vehicle/Walk-in':
            continue
        else:
            writer.writerow(row)

with open('fall2.csv', 'rb') as inp, open('fall3.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[3].strip() == "M":
            row[3] = "1"
            writer.writerow(row)
        elif row[3].strip() == "F":
            row[3] = "0"
            writer.writerow(row)
        else:
            writer.writerow(row)
with open('/fall3.csv', 'rb') as inp, open('fall4.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[21].strip() == "ssisted Respiratory Rate" or row[21].strip() == "Assisted Respiratory Rate":
            row[21] = "1"
            writer.writerow(row)
        else:
            writer.writerow(row)
