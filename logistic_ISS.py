import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#Function to train the logitic regression model and compute the accuracy on the test set and also cross vaildate using k fold.
def logistic(X,Y):
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit()
    # print(result.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    logreg = LogisticRegression(penalty='l2',class_weight='balanced')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    modelCV = LogisticRegression(penalty='l1',class_weight='balanced')
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
    print "Model Validation values:"
    print(classification_report(y_test, y_pred))
    Confusion_matrix = confusion_matrix(y_test, y_pred)
    print "Confusion Matrix:"
    print(Confusion_matrix)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print "Area Under ROC curve:"
    print metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % metrics.auc(fpr,tpr))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

# Logistic on ISS

with open('edittedTrauma.csv', 'rb') as inp, open('ISS.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[40].strip() == '*NA' or row[40].strip() == '':
            continue
        else:
            writer.writerow(row)

with open('ISS.csv', 'rb') as inp, open('ISS1.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4].strip() == "2":
            row[4] = "0"
            writer.writerow(row)
        else:
            writer.writerow(row)

fields = ['Injury Severity Score']
X = pd.read_csv(
    filepath_or_buffer='ISS1.csv',
    usecols=fields)

fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='ISS1.csv',
    usecols=fields1)

print "Efficiency on Logistic Regression Model:"
logistic(X,Y)
