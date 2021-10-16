import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection


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


def Human_HA():
    seqN = []
    seq = readSeq('HUMAN-HA.txt')
    for i in seq:
        seqN.append(i)
        
    mydict = {'Proteins':"HUMAN_HA", 'Sequences':seqN}
    return pd.DataFrame(mydict)
ha = Human_HA()
def Human_NA():
    seqN = []
    seq = readSeq('HUMAN-NA.txt')
    for i in seq:
        seqN.append(i)
        
    mydict = {'Proteins':"HUMAN_NA", 'Sequences':seqN}
    return pd.DataFrame(mydict)
na = Human_NA()
df_row = pd.concat([ha, na], ignore_index=True)
print(df_row)


# In[2]:


fig = plt.figure(figsize=(8,6))
df_row.groupby('Proteins').Sequences.count().plot.bar(ylim=0)
plt.show()


# In[3]:


def getNGram(sequence, size=7):
    return [sequence[i:i+size].lower() for i in range(len(sequence) - size + 1)]

df_row['words'] = df_row.apply(lambda x: getNGram(x['Sequences']), axis=1)
df_row = df_row.drop('Sequences', axis=1)
print(df_row)


# In[4]:


df_row.loc[df_row["Proteins"]=='HUMAN_HA', "Proteins"]= 0
df_row.loc[df_row["Proteins"]=='HUMAN_NA', "Proteins"]= 1
print(df_row)


# In[5]:


df = list(df_row['words'])
for item in range(len(df)):
    df[item] =' '.join(df[item])
# print(df[0])


# In[6]:


df_y = df_row.iloc[:, 0].values


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_digits

X_vector = CountVectorizer(ngram_range=(4,4))
x_traincv = X_vector.fit_transform(df)


X_train, X_test, y_train, y_test = train_test_split(x_traincv, df_y, test_size = 0.20, random_state=1)

#a = x_traincv.toarray()
# print(a)

print(X_train.shape)
y_train = y_train.astype('int')


# In[8]:


classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[9]:


from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
from itertools import cycle
# #X_testcv = X_vector.transform(X_test)
y_pred = classifier.predict(X_test)
print(y_pred)

y_test = y_test.astype('int')
print("Training set score: {:.3f}".format(classifier.score(X_train, y_train)))
print("Testing set score: {:.3f}".format(classifier.score(X_test, y_test)))
print('**********************************************************************************')
print()
print()
print('Balanced Parameters')
print(np.unique(y_pred, return_counts=True))
print(np.unique(y_test, return_counts=True))
print()
print('Confusion Matrix')
print()
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def conf_matx(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = conf_matx(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# In[10]:


titles = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles:
    viewMatrix = plot_confusion_matrix(classifier, X_test, y_test,
                                 
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    viewMatrix.ax_.set_title(title)

    #print(title)
    print(viewMatrix.confusion_matrix)

plt.show()


# In[11]:


print(classification_report(y_test, y_pred))


# In[12]:


from sklearn.metrics import roc_auc_score

all_zreos = [0 for _ in range(len(y_test))]

# calculate scores
ns_auc = roc_auc_score(y_test, all_zreos)
lr_auc = roc_auc_score(y_test, y_pred)
# summarize scores
print('HUMAN-HA: ROC AUC=%.3f' % (ns_auc))
print('HUMAN-NA: ROC AUC=%.3f' % (lr_auc))
# determine the roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, all_zreos)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
# graph the roc curve
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='HUMAN-HA')
plt.plot(lr_fpr, lr_tpr, marker='.', label='HUMAN-NA')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
