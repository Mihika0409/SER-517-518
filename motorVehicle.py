import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

fields = ['Age in Years', 'Gender', 'MV Speed','RTS','Field GCS','Field SBP', 'Field HR', 'Field RR','Intubated','Resp Assistance']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/m4.csv',
    usecols=fields)
fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/m4.csv',
    usecols=fields1)
df['RTS'] = df['RTS'].replace(['*NA','*ND','*BL',''],'7.65')
#df['RTS'] = np.where(((df['RTS'] == '*NA') | (df['RTS'] == '*ND') | (df['RTS'] == '*BL') | (df['RTS'] == '')) & (df['Gender'] == 1),'6.15',df['RTS'])
#df['RTS'] = np.where(((df['RTS'] == '*NA') | (df['RTS'] == '*ND') | (df['RTS'] == '*BL') | (df['RTS'] == '')) & (df['Gender'] == 0),'7.75',df['RTS'])
#df['Field GCS'] = np.where(((df['Field GCS'] == '*NA') | (df['Field GCS'] == '*ND') | (df['Field GCS'] == '*BL') | (df['Field GCS'] == '')) & (df['Gender'] == 1),'10.24',df['Field GCS'])
#df['Field GCS'] = np.where(((df['Field GCS'] == '*NA') | (df['Field GCS'] == '*ND') | (df['Field GCS'] == '*BL') | (df['Field GCS'] == '')) & (df['Gender'] == 0),'14.90',df['Field GCS'])
#df['Field SBP'] = np.where(((df['Field SBP'] == '*NA') | (df['Field SBP'] == '*ND') | (df['Field SBP'] == '*BL') | (df['Field SBP'] == '')) & (df['Gender'] == 1),'111.50',df['Field SBP'])
#df['Field SBP'] = np.where(((df['Field SBP'] == '*NA') | (df['Field SBP'] == '*ND') | (df['Field SBP'] == '*BL') | (df['Field SBP'] == '')) & (df['Gender'] == 0),'120.25',df['Field SBP'])
#df['Field HR'] = np.where(((df['Field HR'] == '*NA') | (df['Field HR'] == '*ND') | (df['Field HR'] == '*BL') | (df['Field HR'] == '')) & (df['Gender'] == 1),'106',df['Field HR'])
#df['Field HR'] = np.where(((df['Field HR'] == '*NA') | (df['Field HR'] == '*ND') | (df['Field HR'] == '*BL') | (df['Field HR'] == '')) & (df['Gender'] == 0),'110',df['Field HR'])
df['Field GCS']=df['Field GCS'].replace(['*NA','*ND','*BL',''],'14.54')
df['Field SBP']=df['Field SBP'].replace(['*NA','*ND','*BL',''],'119')
df['Field HR']=df['Field HR'].replace(['*NA','*ND','*BL',''],'110')
df['Field RR']=df['Field RR'].replace(['*NA','*ND','*BL',''],'21')
df['MV Speed'] = df['MV Speed'].replace(['*NA','*ND','*BL',''],'0')


#Function to train the logitic regression model and compute the accuracy on the test set and also cross vaildate using k fold.
def logistic(X,Y):
    # print np.asarray(X)
    logit_model = sm.Logit(Y, X.astype(float))
    result = logit_model.fit()
    # print(result.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    logreg = LogisticRegression(penalty='l2',class_weight='balanced')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    modelCV = LogisticRegression(penalty='l2',class_weight='balanced',fit_intercept = False)
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
    
# Function to train the Decision Trees Classifier and compute the accuracy on the test set and also cross vaildate using k fold.
def decisionClassifier(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    clf = DecisionTreeClassifier(criterion="gini" , presort=True, random_state=42, class_weight='balanced', splitter='best', max_depth=10, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yTrain_pred = clf.predict(X_train)
    print "Test Accuracy is ", accuracy_score(y_test, y_pred)
    print "Training Accuracy is ", accuracy_score(y_train, yTrain_pred)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    modelCV = DecisionTreeClassifier(criterion="gini", presort=True, random_state=42, class_weight='balanced', splitter='best', max_depth=10, min_samples_leaf=1)
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
    Confusion_matrix = confusion_matrix(y_test, y_pred)
    print "Confusion Matrix:"
    print(Confusion_matrix)
    
print "Efficiency on Logistic Regression Model:"
logistic(df,Y)

print "Efficiency on Decision Tree Classifier Model:"
decisionClassifier(df,Y)
