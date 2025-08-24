#Sample Code:
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
# Model definition
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training
history = model.fit(X, Y, epochs=1000, verbose=0)
# Evaluate
loss, acc = model.evaluate(X, Y)
print("Accuracy:", acc)
# Predict
print("Predictions:", model.predict(X))
# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
#output(screenshot):

<img width="1920" height="1080" alt="Screenshot 2025-08-24 105135" src="https://github.com/user-attachments/assets/b17bac46-60c4-4025-82c0-59621924698b" />
<img width="1920" height="1080" alt="Screenshot 2025-08-24 105155" src="https://github.com/user-attachments/assets/c34cf950-8424-4a58-8c29-e072ee6d5931" />


