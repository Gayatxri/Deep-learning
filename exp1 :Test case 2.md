#samplecode
import numpy as np
from sklearn.linear_model import Perceptron

# Test Case Set 2: Perceptron Prediction
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X)

print("=== Test Case Set 2: Perceptron Prediction ===")
for xi, yi, yp in zip(X, y, y_pred):
    remark = "Correct" if yi == yp else "May fail"
    print(f"Input: {xi}, Predicted: {yp}, Expected: {yi}, Remark: {remark}")
    #output(screenshot)
    <img width="1920" height="1080" alt="Screenshot 2025-08-24 141353" src="https://github.com/user-attachments/assets/4043dfef-5533-419d-a54b-d6637e48d938" />

