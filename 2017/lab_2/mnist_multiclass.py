import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

# read data
mnist = fetch_mldata("MNIST original")
X_train, Y_train = mnist.data[:60000] / 255., mnist.target[:60000]
X_test, Y_test = mnist.data[60000:] / 255., mnist.target[60000:]
m, d = X_train.shape;  # m number of examples, d instance dimesion
k = max(Y_train)+1  # k number of classes

# show some images
plt.figure(1);
for i in range (1,26):
    ax = plt.subplot(5,5,i);
    ax.axis('off');
    ax.imshow(255-X_train[i,:].reshape(28,28),cmap="gray");
plt.show();

# weight vector
w = np.zeros((k, d));
M = 0; # counts mistakes
eta = 0.1 # learning rate

# multiclass perceptron
epochs = 10
for e in range(epochs):	
	X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
	for x, y in zip(X_train, Y_train):		    		
	   	# predict
	   	y_hat = np.argmax(np.dot(w, x))    
	    # update
		if y != y_hat:
			w[y, :] = w[y, :] + eta * x
			w[y_hat, :] = w[y_hat, :] - eta * x	

# show the mask learnt by multiclass preceptron
plt.figure(2);
classes = np.arange(10)
for i, c in enumerate(classes):    
    ax1 = plt.subplot(2, 5, i+1);
    ax1.axis('off');
    ax1.title.set_text(str(c))
    tmp = 1/(1 + np.exp(-10 * w[c,: ] / w[c,: ].max()));
    ax1.imshow(tmp.reshape(28,28),cmap="gray");
plt.show();
#%%


# check performence on test data
m, d = X_test.shape;
M = 0
for i in range(0, m):
    y_hat = np.argmax(np.dot(w, X_test[i,:]))
    if Y_test[i] != y_hat:
        M = M + 1
print "test err=", float(M)/m

