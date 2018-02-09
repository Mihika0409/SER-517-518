import pandas as pd
import numpy as np
import scipy.stats as sp
import csv
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from plistlib import Data

#Will only consider level 1 and level 2 trauma.
with open('/Users/satishnandan/Desktop/TraumaActivation/traumaNew.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/edittedTrauma.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4].strip() == "1" or row[4].strip() == "2" or row[4].strip() == "Levels":
            writer.writerow(row)

#Will remove null values from the mv speed column.
with open('/Users/satishnandan/Desktop/TraumaActivation/edittedTrauma.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma1.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[13].strip() == "*NA" or row[13].strip() == "*ND" or row[13].strip() == "*BL":
            continue
        else:
            writer.writerow(row)

#Will ignore private/ public vehicle/ walkin cases.
with open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma1.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[14].strip() == "Private/Public Vehicle/Walk-in":
            continue
        else:
            writer.writerow(row)

# Will modify the gender column into binary data.
with open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma2.csv', 'wb') as out:
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

# Will modify the assisted Respitaroy column to numerical data.
with open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma2.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma3.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[21].strip() == "ssisted Respiratory Rate" or row[21].strip() == "Assisted Respiratory Rate":
            row[21] = "1"
            writer.writerow(row)
        else:
            writer.writerow(row)

#Will input 0 to null fields.
with open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma3.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma4.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[22].strip() == "":
            row[22] = "0"
            writer.writerow(row)
        else:
            writer.writerow(row)

#Will convert the trauma level to binary.
with open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma4.csv', 'rb') as inp, open('/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4].strip() == "2":
            row[4] = "0"
            writer.writerow(row)
        else:
            writer.writerow(row)

fields = ['Age in Years', 'Gender', 'MV Speed','Resp Assistance','RTS','Field GCS','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR']
fieldsPCA = ['Age in Years', 'Gender', 'MV Speed','Resp Assistance','RTS']

pca = ['Levels','Field GCS','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv',
    usecols=fields)
df1 = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv',
    usecols=fieldsPCA)
dfPCA = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv',
    usecols=pca)
fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv',
    usecols=fields1)

dfPCA.dropna(how="all", inplace=True)
# print dfPCA.tail()

X = dfPCA.ix[:,1:5].values
y = dfPCA.ix[:,0].values

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
# print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
# print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])

sklearn_pca = sklearnPCA(n_components=3)
Y_sklearn = sklearn_pca.fit_transform(X_std)
# print Y_sklearn

sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
# print Y_sklearn

#Create the data frame and combine it into one data frame.
dfPCA = pd.DataFrame(Y_sklearn)
dfNew = pd.concat([df1,dfPCA],axis=1)


#Function to train the logitic regression model and compute the accuracy on the test set and also cross vaildate using k fold.
def logistic(X,Y):
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit()
    # print(result.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


#Function to train the Decision Trees Classifier and compute the accuracy on the test set and also cross vaildate using k fold.
def decisionClassifier(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
    clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(X_test)
    print "Accuracy is ", accuracy_score(y_test, y_pred)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = DecisionTreeClassifier()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


print "Efficiency on Logistic Regression Model before PCA:"
logistic(df,Y)
print "Efficiency on Logistic Regression Model after PCA:"
logistic(dfNew,Y)
print "Efficiency on Decision Tree Classifier Model before PCA:"
decisionClassifier(df,Y)
print "Efficiency on Decision Tree Classifier Model after PCA:"
decisionClassifier(dfNew,Y)
