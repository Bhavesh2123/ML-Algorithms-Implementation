# Now let us start with Stochastic Gradient descent 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#Generating data
np.random.seed(42)
X = 2* np.random.rand(100, 1)
y= 4 + 3*X +np.random.randn(100, 1)
# For a Linear regression with one feature we use
# y = b0 + b1X
# Now we will define SGD
def sgd(X,y ,learning_rate=0.1, epochs=1000, batch_size=1):
    m=len(X)
    theta= np.random.randn(2,1)
    X_bias = np.c_[np.ones((m,1)),X]
    cost_historyy=[]

    for epoch in range(epochs):
        indices= np.random.permutation(m)
        X_shuffeled =X_bias[indices]
        y_shuffeled= y[indices]

        for i in range(0, m , batch_size):
            X_batch = X_shuffeled[i:i+batch_size]
            y_batch = y_shuffeled[i:i+batch_size]

            gradient= 2 / batch_size * X_batch.T.dot(X_batch.dot(theta)-y_batch)
            theta= theta-learning_rate*gradient

        predictions= X_bias.dot(theta)
        cost = np.mean((predictions - y )** 2)
        cost_historyy.append(cost)

        if epoch % 100 ==0:
            print(f"Epoch {epoch}, Cost: {cost}")

    return theta, cost_historyy

theta_final, cost_history = sgd(X, y, learning_rate=0.1, epochs=1000, batch_size=1)
           
# Now plotting the graphs 
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function during Training')
plt.show()