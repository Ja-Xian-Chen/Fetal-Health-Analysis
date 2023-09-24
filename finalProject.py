import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv("fetal_health.csv")

# Present a visual distribution of the 3 classes. Is the data balanced?
# How do you plan to circumvent the data imbalance problem, if there is one?

count = {}

y = data['fetal_health']

for i in y:
    if i in count:
        count[i] += 1
    else:
        count[i] = 1

xAxis = ['Normal', 'Suspect', 'Pathological']
yAxis = [count[1], count[2], count[3]]

plt.ylabel("Count")
plt.bar(xAxis, yAxis)
plt.show()

# Present 10 features that are most reflective to fetal health conditions
# (there are more than one way of selecting features and any of these are acceptable).
# Present if the correlation is statistically significant (using 95% and 90% critical values).

correlation = {}
x = data.iloc[:, :-1]
values = []
for features in x.columns:
    values = stats.pearsonr(data[features], y)
    coEfficient = values[0]
    pValue = values[1]
    correlation[features] = (features, abs(coEfficient))
    if pValue <= 0.1:
        print(features, "is Significant at", 0.9)
    else:
        print(features, "is not Significant at", 0.9)
    if pValue <= 0.05:
        print(features, "is Significant at", 0.95)
    else:
        print(features, "is not Significant at", 0.95)
    print("")
print("10 Features")
b = sorted(sorted(correlation, key=lambda x: x[0]), key=lambda x: x[1], reverse=True)
count = 0
for i in b:
    if count < 10:
        print(i)
    count = count + 1

# Develop two different models to classify CTG features into the three fetal health states
# (I intentionally did not name which two models.
# Note that this is a multiclass problem that can also be treated as regression, since the labels are numeric.)
#Model 1
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=27)

dt = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
dt.fit(X_train, y_train)
y_pred = dt.predict(x_test)

print("\nDecision Tree:")
figure = plt.figure(figsize=(15, 10))
plot = tree.plot_tree(dt)
textRepresentation = tree.export_text(dt)
print(textRepresentation)
# Model 2(error)
#X = data['uterine_contractions'].values.reshape(-1,1)
#Y = data['fetal_health'].values.reshape(-1,1)
#x_train,x_test,Y_train,y_test=train_test_split(X,Y)

#regressor = LinearRegression()
#regressor.fit(x_train, Y_train)
#y_pred = regressor.predict(X_test)

#plt.scatter(x_train, y_train, color='g')
#plt.plot(x_test, y_pred, color='k')
#plt.show()

# Visually present the confusion matrix
plot_confusion_matrix(dt, x_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# With a testing set of size of 30% of all available data, calculate
#   Area under the ROC Curve
#   F1 Score
#   Area under the Precision-Recall Curve
#   (for both models in 3)
print("")
print("Precision: ", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1: ", f1_score(y_test, y_pred, average='macro'))
auc = roc_auc_score(y_test, dt.predict_proba(x_test), multi_class='ovo')
print('Area under ROC curve:', auc)
print("")

# Without considering the class label attribute,
# use k-means clustering to cluster the records in different clusters and visualize them
# (use k to be 5, 10, 15).

m = 7
n = 12
# 5
cluster = x.iloc[:, [m, n]]
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(cluster)
pred_y = kmeans.fit_predict(cluster)

plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], c=pred_y, cmap=plt.cm.Paired)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.title(str(5) + " Clusters")
plt.xlabel(x.columns[m])
plt.ylabel(x.columns[n])
plt.show()
# 10
cluster = x.iloc[:, [m, n]]
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(cluster)
pred_y = kmeans.fit_predict(cluster)

plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], c=pred_y, cmap=plt.cm.Paired)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.title(str(10) + " Clusters")
plt.xlabel(x.columns[m])
plt.ylabel(x.columns[n])
plt.show()
# 15
cluster = x.iloc[:, [m, n]]
kmeans = KMeans(n_clusters=15, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(cluster)
pred_y = kmeans.fit_predict(cluster)

plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], c=pred_y, cmap=plt.cm.Paired)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.title(str(15) + " Clusters")
plt.xlabel(x.columns[m])
plt.ylabel(x.columns[n])
plt.show()
