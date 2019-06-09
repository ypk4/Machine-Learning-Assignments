# Iris dataset flower type recognition using SVM

from sklearn import datasets, metrics
from sklearn.svm import SVC

train_size = 0.7

iris = datasets.load_iris()

x = iris.data
y = iris.target

nSamples = len(x)
x_train = x[: int(nSamples * train_size)]
y_train = y[: int(nSamples * train_size)]

svc = SVC(kernel="linear", C=1)

svm = svc.fit(x_train, y_train)		# Training

actual = y[int(nSamples * train_size) :]
predicted = svc.predict(x[int(nSamples * train_size) :])						# Prediction

print("Classification report %s : \n%s\n" %(svm, metrics.classification_report(actual, predicted)))

print("Confusion matrix: %s" %metrics.confusion_matrix(actual, predicted))

print("\nAccuracy score = ", metrics.accuracy_score(actual, predicted), '\n')
