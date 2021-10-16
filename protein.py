import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
# %matplotlib inline


def readSeq(filename):

    with open(filename, 'r') as f:
        data = ''
        name_list = []
        seq_list = []

        for line in f:

            line = line.rstrip()
            for i in line:
                if i == '>':
                    name_list.append(line)
                    if data:
                        seq_list.append(data)
                        data = ''
                    break
                else:
                    line = line.upper()
            if all([k == k.upper() for k in line]):
                data = data + line
    return seq_list


def printHuman1():
    seqN = []
    seq = readSeq('H1N1.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "H1N1", 'Sequences': seqN}
    return pd.DataFrame(mydict)


h1n1 = printHuman1()


def printHuman2():
    seqN = []
    seq = readSeq('H2N2.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "H2N2", 'Sequences': seqN}
    return pd.DataFrame(mydict)


h2n2 = printHuman2()


def printHuman3():
    seqN = []
    seq = readSeq('H3N2.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "H3N2", 'Sequences': seqN}
    return pd.DataFrame(mydict)


h3n2 = printHuman3()


def printHuman5():
    seqN = []
    seq = readSeq('H5N1.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "H5N1", 'Sequences': seqN}
    return pd.DataFrame(mydict)


h5n1 = printHuman5()


def printAvian1():
    seqN = []
    seq = readSeq('Av-H1N1.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "Av-H1N1", 'Sequences': seqN}
    return pd.DataFrame(mydict)


avian1 = printAvian1()


def printAvian2():
    seqN = []
    seq = readSeq('Av-H2N2.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "Av-H2N2", 'Sequences': seqN}
    return pd.DataFrame(mydict)


avian2 = printAvian2()


def printAvian3():
    seqN = []
    seq = readSeq('Av-H3N2.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "Av-H3N2", 'Sequences': seqN}
    return pd.DataFrame(mydict)


avian3 = printAvian3()


def printAvian5():
    seqN = []
    seq = readSeq('Av-H5N1.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "Av-H5N1", 'Sequences': seqN}
    return pd.DataFrame(mydict)


avian5 = printAvian5()


def printSwine1():
    seqN = []
    seq = readSeq('SW-H1N1.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "SW-H1N1", 'Sequences': seqN}
    return pd.DataFrame(mydict)


swine1 = printSwine1()


def printSwine3():
    seqN = []
    seq = readSeq('SW-H3N2.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "SW-H3N2", 'Sequences': seqN}
    return pd.DataFrame(mydict)


swine3 = printSwine3()


def printSwine5():
    seqN = []
    seq = readSeq('SW-H5N1.txt')
    for i in seq:
        seqN.append(i)

    mydict = {'Serotype': "SW-H5N1", 'Sequences': seqN}
    return pd.DataFrame(mydict)


swine5 = printSwine5()
df_row = pd.concat([h1n1, h2n2, h3n2, h5n1, swine1, swine3, swine5, avian1, avian2, avian3, avian5], ignore_index=True)
print(df_row)
counts = df_row.Serotype.value_counts()
print(counts)

# plot counts
# plt.figure()
# sns.distplot(counts, hist = False, color = 'purple')
# plt.title('Count Distribution for Serotpes Types')
# plt.ylabel('% of records')
# plt.show()
fig = plt.figure(figsize=(8, 6))
df_row.groupby('Serotype').Sequences.count().plot.bar(ylim=0)
plt.show()

def getNGram(sequence, size=7):
    return [sequence[i:i+size] for i in range(len(sequence) - size + 1)]

df_row['words'] = df_row.apply(lambda x: getNGram(x['Sequences']), axis=1)
df_row = df_row.drop('Sequences', axis=1)
print(df_row)

df_row.loc[df_row["Serotype"]=='H1N1', "Serotype"]= 0
df_row.loc[df_row["Serotype"]=='H2N2', "Serotype"]= 1
df_row.loc[df_row["Serotype"]=='H3N2', "Serotype"]= 2
df_row.loc[df_row["Serotype"]=='H5N1', "Serotype"]= 3
df_row.loc[df_row["Serotype"]=='SW-H1N1', "Serotype"]= 4
df_row.loc[df_row["Serotype"]=='SW-H3N2', "Serotype"]= 5
df_row.loc[df_row["Serotype"]=='SW-H5N1', "Serotype"]= 6
df_row.loc[df_row["Serotype"]=='Av-H1N1', "Serotype"]= 7
df_row.loc[df_row["Serotype"]=='Av-H2N2', "Serotype"]= 8
df_row.loc[df_row["Serotype"]=='Av-H3N2', "Serotype"]= 9
df_row.loc[df_row["Serotype"]=='Av-H5N1', "Serotype"]= 10
print(df_row)

df = list(df_row['words'])
for item in range(len(df)):
    df[item] =' '.join(df[item])
# print(df[0])

# df_X = df
df_Y = df_row.iloc[:, 0].values
# print(df_Y)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

X_vector = CountVectorizer(ngram_range=(4,4))


X = X_vector.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X, df_Y, test_size = 0.20, random_state=1)

print(X_train.shape)
y_train = y_train.astype('int')

classifier = MultinomialNB()
# cross validation
y = df_Y.astype('int')
scores = cross_val_score(classifier, X, y, cv=10, scoring='accuracy')
print(scores)
print('Average accuracy using MultinomialNB: ', scores.mean())

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
scoreRF = cross_val_score(rf, X, y, cv=10, scoring='accuracy')
print(scoreRF)
print('Average accuracy using RandomForestClassifier: ', scoreRF.mean())

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
scoreDT = cross_val_score(DT, X, y, cv=10, scoring='accuracy')
print(scoreDT)
print('Average accuracy using DecisionTreeClassifier: ', scoreDT.mean())

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
scoreKNN = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scoreKNN)
print('Average accuracy using KNeighborsClassifier: ', scoreKNN.mean())

models = []
models.append(('KNN', scoreKNN))
models.append(('DT', scoreDT))
models.append(('NB', scores))
models.append(('RFC', scoreRF))
# models.append(('ADA', scoreADA))
print(models)

results = []
names = []

for name, model in models:
    results.append(model)
    names.append(name)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from itertools import cycle


y_test = y_test.astype('int')
print("Training set score: {:.3f}".format(classifier.score(X_train, y_train)))
print("Testing set score: {:.3f}".format(classifier.score(X_test, y_test)))
print('**********************************************************************************')
print()
print()
# print('Balanced Parameters')
# print(np.unique(y_pred, return_counts=True))
# print(np.unique(y_test, return_counts=True))
print()
print('Confusion Matrix')
print()
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def conf_matx(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = conf_matx(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
matrix = matrix.astype('float')/ matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
sns.heatmap(matrix)

print(classification_report(y_test, y_pred))

from sklearn.preprocessing import label_binarize

y_pred = classifier.fit(X_train, y_train).predict_proba(X_test)
y_test1 = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9,10])
n_classes = y_test1.shape[1]

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class using "fall positive rate" and "true positive rate"
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'pink', 'red', 'orange', 'purple', 'yellow', 'green', 'navy'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of different types of Serotypes')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
from itertools import cycle


#y_test = y_test.astype('int')
print("Training set score: {:.3f}".format(rf.score(X_train, y_train)))
print("Testing set score: {:.3f}".format(rf.score(X_test, y_test)))
print('**********************************************************************************')
print()
print()
# print('Balanced Parameters')
# print(np.unique(y_pred, return_counts=True))
# print(np.unique(y_test, return_counts=True))
print()
print('Confusion Matrix')
print()
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred_rf, name='Predicted')))
def conf_matx(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = conf_matx(y_test, y_pred_rf)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

matrix2 = confusion_matrix(y_test, y_pred_rf)
print(matrix2)
matrix2 = matrix2.astype('float')/ matrix2.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
sns.heatmap(matrix2)

from sklearn.preprocessing import label_binarize

y_pred = rf.fit(X_train, y_train).predict_proba(X_test)
y_test2 = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9,10])
n_classes = y_test2.shape[1]

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class using "fall positive rate" and "true positive rate"
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'pink', 'red', 'orange', 'purple', 'yellow', 'green', 'navy'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of different types of Serotypes')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
DT.fit(X_train, y_train)
y_pred_DT = DT.predict(X_test)
from itertools import cycle


#y_test = y_test.astype('int')
print("Training set score: {:.3f}".format(DT.score(X_train, y_train)))
print("Testing set score: {:.3f}".format(DT.score(X_test, y_test)))
print('**********************************************************************************')
print()
print()
# print('Balanced Parameters')
# print(np.unique(y_pred, return_counts=True))
# print(np.unique(y_test, return_counts=True))
print()
print('Confusion Matrix')
print()
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred_DT, name='Predicted')))
def conf_matx(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = conf_matx(y_test, y_pred_DT)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
from itertools import cycle

#y_test = y_test.astype('int')
print("Training set score: {:.3f}".format(knn.score(X_train, y_train)))
print("Testing set score: {:.3f}".format(knn.score(X_test, y_test)))
print('**********************************************************************************')
print()
print()
# print('Balanced Parameters')
# print(np.unique(y_pred, return_counts=True))
# print(np.unique(y_test, return_counts=True))
print()
print('Confusion Matrix')
print()
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred_knn, name='Predicted')))
def conf_matx(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = conf_matx(y_test, y_pred_knn)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
