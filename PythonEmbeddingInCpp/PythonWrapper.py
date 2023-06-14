from sklearn import tree


def TreeFit(xData, labelData,):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(xData, labelData)
    #Возвращаем обученную ML-модель, с которой дальше будем работать
    return clf

