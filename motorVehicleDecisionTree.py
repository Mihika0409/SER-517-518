import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
    
fields = ['Age in Years', 'Gender', 'MV Speed','RTS','Field GCS','Field SBP', 'Field HR', 'Field RR']
df = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeed10.csv',
    usecols=fields)
fields1 = ['Levels']
Y = pd.read_csv(
    filepath_or_buffer='/Users/satishnandan/Desktop/TraumaActivation/mvSpeed10.csv',
    usecols=fields1)


print "Efficiency on Decision Tree Classifier Model:"
decisionClassifier(df,Y)
