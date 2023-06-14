from sklearn import tree

def TreeFit(xData, labelData):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(xData, labelData)
    return clf