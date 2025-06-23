#Hand writing detecting
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(x_test[:10])

# Create HTML report
html = '<html><head><title>MNIST Predictions</title></head><body>'
html += '<h2>MNIST CNN Predictions</h2><table><tr>'

for i in range(10):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    html += f'<td style="text-align:center;"><img src="data:image/png;base64,{img_str}" /><br/>Predicted: <b>{np.argmax(predictions[i])}</b></td>'

html += '</tr></table></body></html>'

# Save to HTML file
with open("mnist_predictions.html", "w") as f:
    f.write(html)

print("HTML file 'mnist_predictions.html' has been created.")