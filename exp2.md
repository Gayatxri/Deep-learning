#samplecode
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255
X_test = X_test.reshape(-1,28,28,1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Build model
model = Sequential([
 Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
 MaxPooling2D(pool_size=(2,2)),
 Flatten(),
 Dense(128, activation='relu'),
 Dense(10, activation='softmax')
])
# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, 
validation_data=(X_test, y_test))
<img width="1920" height="1080" alt="Screenshot 2025-08-27 203507" src="https://github.com/user-attachments/assets/dc754312-5725-441a-a5a3-fc49709655e2" />
<img width="1920" height="1080" alt="Screenshot 2025-08-27 203458" src="https://github.com/user-attachments/assets/5eddd99b-2755-4809-b95a-467a910296f9" />
