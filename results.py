X= df.iloc[:,:-1].values
y= df['Levels'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)

#y_train_pred = dt.predict(X_train)
#y_test_pred = dt.predict(X_test)

#print('MSE train: %.3f, test: %.3f' % ( mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))


#decision trees with the gini index
#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred_train = clf_gini.predict(X_train)

print y_pred

print "test Accuracy is ", accuracy_score(y_test,y_pred)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train))*100

# decision tree with the information gain
#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_train_en = clf_gini.predict(X_train)

print y_pred_en

#print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100

print "test Accuracy is ", accuracy_score(y_test,y_pred_en)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train_en)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred_en))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train_en))*100


def logistic(X,Y):
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit()
    # print(result.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    logreg = LogisticRegression(penalty='l1',class_weight='balanced')
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
