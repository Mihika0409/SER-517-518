import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

fields = ['Age in Years', 'Gender', 'MV Speed','Resp Assistance','RTS','Field GCS','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv',
    usecols=fields)
fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeedTrauma5.csv',
    usecols=fields1)

def logistic(X,Y):
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit()
    # print(result.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    logreg = LogisticRegression(penalty='l1',class_weight='balanced')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    modelCV = LogisticRegression(penalty='l1',class_weight='balanced')
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
    print(classification_report(y_test, y_pred))
    Confusion_matrix = confusion_matrix(y_test, y_pred)
    print(Confusion_matrix)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
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



print "Efficiency and ROC Graph on Logistic Regression Model:"
logistic(df,Y)
