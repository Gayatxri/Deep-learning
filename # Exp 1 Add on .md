#Sample Code :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
# Plotting
for i in range(len(X)):
 if y[i] == 0:
 plt.scatter(X[i][0], X[i][1], color='red')
 else:
 plt.scatter(X[i][0], X[i][1], color='blue')
# Decision boundary
x_values = [0, 1]
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()
#output(screenshot):
<img width="1920" height="1080" alt="Screenshot 2025-08-24 114950" src="https://github.com/user-attachments/assets/86fac126-3982-420b-a8c1-86568a4d433c" />
<img width="1920" height="1080" alt="Screenshot 2025-08-24 114938" src="https://github.com/user-attachments/assets/00732f21-1a96-4b57-9f30-9f5f9721075e" />

