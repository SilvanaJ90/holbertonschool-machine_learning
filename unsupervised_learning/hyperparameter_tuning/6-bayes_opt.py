#!/usr/bin/env python3
""" Hyperparameter optimization for a neural network using Bayesian Optimization """

import numpy as np
import GPyOpt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size, epochs):
    model = Sequential()
    model.add(Dense(num_units, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size):
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

def objective_function(params):
    learning_rate, num_units, dropout_rate, l2_weight, batch_size, epochs = params[0]
    num_units = int(num_units)
    batch_size = int(batch_size)
    epochs = int(epochs)

    model = create_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size, epochs)
    history = train_model(model, epochs, batch_size)
    val_acc = max(history.history['val_accuracy'])
    
    # Apply k-NN for clustering
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    X_train_features = feature_extractor.predict(x_train)
    X_test_features = feature_extractor.predict(x_test)

    k = 10  # Number of clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train_features)
    train_cluster_labels = kmeans.labels_
    test_cluster_labels = kmeans.predict(X_test_features)

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_features, train_cluster_labels)

    predicted_test_cluster_labels = knn_classifier.predict(X_test_features)

    accuracy = accuracy_score(test_cluster_labels, predicted_test_cluster_labels)
    print("Accuracy of clustering with k-NN:", accuracy)
    
    return -val_acc  # Minimize negative accuracy

# Define the bounds of the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (50, 200)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'epochs', 'type': 'discrete', 'domain': (5, 10, 15)}
]

# Run Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=bounds,
    initial_design_numdata=5,
    acquisition_type='EI',
    acquisition_jitter=0.05,
    maximize=True
)
optimizer.run_optimization(max_iter=30)

# Save optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write("Optimization Results:\n")
    f.write(f"Best Parameters: {optimizer.x_opt}\n")
    f.write(f"Best Objective: {-optimizer.fx_opt}\n")
