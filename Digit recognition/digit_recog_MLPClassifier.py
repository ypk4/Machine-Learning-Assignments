# Digit recognition using MLPClassifier
# A multilayer perceptron (MLP) is a class of feedforward artificial neural network. A MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier

train_size = 0.7

input_images = datasets.load_digits()
# input_images --> images, data, target, target_names, desc(description)
nSamples = len(input_images.images)

inData = input_images.images.reshape((nSamples, -1))

classifier = MLPClassifier(hidden_layer_sizes = (30,30,30))
classifier.fit(inData[: int(nSamples * train_size)], input_images.target[: int(nSamples * train_size)])			# Training

actual = input_images.target[int(nSamples * train_size) :]
predicted = classifier.predict(inData[int(nSamples * train_size) :])						# Prediction

print("Classification report %s : \n%s\n" %(classifier, metrics.classification_report(actual, predicted)))

print("Confusion matrix: %s" %metrics.confusion_matrix(actual, predicted))

print("\nAccuracy score = ", metrics.accuracy_score(actual, predicted), '\n')
