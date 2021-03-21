#!/usr/bin/env python
# coding: utf-8

# # Breast Tissue Data Set-UCI ML
# #Source:
# 
# JP Marques de Sá, INEB-Instituto de Engenharia Biomédica, Porto, Portugal; e-mail: jpmdesa '@' gmail.com
# J Jossinet, inserm, Lyon, France

# # Logistic Regression

# In[2]:


from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv(r'C:\Users\dell\Desktop\BreastTissue.csv')
x=df.drop(['Case #','Class'],axis='columns')
y=df['Class']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x1 = df.apply(le.fit_transform)
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, random_state=0)
y=y.values.reshape(-1,1)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x1_train, y_train)
y_model= logreg.predict(x1_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_model))
print(classification_report(y_test, y_model))
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
mat = confusion_matrix(y_test, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')


# In[10]:


# Finetuning logisticregression model using gridsearch
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x1_train, y_train)
print("training accuracy :", model.score(x1_train, y_train))
print("testing accuracy :", model.score(x1_test, y_test))
# grid search cross validation with 2 hyperparameter
# 1. hyperparameter is C:logistic regression regularization parameter
# 2. penalty l1 or l2
# Hyperparameter grid
from sklearn.model_selection import GridSearchCV
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=3)
logreg_cv.fit(x1_train,y_train)

# Print the optimal parameters and best score
print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))


# # Decision Tree Algorithm

# In[5]:


from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv(r'C:\Users\dell\Desktop\BreastTissue.csv')
x=df.drop(['Case #','Class'],axis='columns')
y=df['Class']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x1 = df.apply(le.fit_transform)
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, random_state=0)
y=y.values.reshape(-1,1)
classifier=DecisionTreeClassifier()
classifier=classifier.fit(x1_train,y_train)
y_model= classifier.predict(x1_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_model))
print(classification_report(y_test, y_model))
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
mat = confusion_matrix(y_test, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# # KNN Algorithm

# In[6]:


from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r'C:\Users\dell\Desktop\BreastTissue.csv')
x=df.drop(['Case #','Class'],axis='columns')
y=df['Class']
le = preprocessing.LabelEncoder()
x1 = df.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(x1)
onehotlabels = enc.transform(x1).toarray()
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, random_state=0)
y=y.values.reshape(-1,1)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(x1_train,y_train)
y_model= classifier.predict(x1_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_model))
print(classification_report(y_test, y_model))
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
mat = confusion_matrix(y_test, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')


# In[9]:


# Determining k value in KNN algorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
df=pd.read_csv(r'C:\Users\dell\Desktop\BreastTissue.csv')
x=df.drop(['Case #','Class'],axis='columns')
y=df['Class']
le = preprocessing.LabelEncoder()
x1 = df.apply(le.fit_transform)
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, random_state=0)
k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(k_values):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x1_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x1_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x1_test, y_test))

    # Plot
plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# In[11]:


# SVM, pre-process and pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x1_train,y_train)

y_pred = cv.predict(x1_test)

print("Accuracy: {}".format(cv.score(x1_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# In[ ]:




