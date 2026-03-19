import tensorflow as tf

train_dir = "dataset/train"
test_dir = "dataset/test"

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(64,64),
    batch_size=16
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(64,64),
    batch_size=16
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, validation_data=test_data, epochs=3)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()