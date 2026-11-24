
import tensorflow as tf
import tensorflow_datasets as tfds

print("Loading EMNIST dataset...")

# Load digits and letters
(train_data, test_data), info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# Keep digits (0-9) and uppercase letters (A-Z)
def keep_selected(image, label):
    return label <= 35


# Correct image orientation and normalize pixels
def prepare_image(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


print("Preparing training dataset...")

train_data = train_data.filter(keep_selected)
train_data = train_data.map(
    prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_data = train_data.shuffle(10000)
train_data = train_data.batch(128)
train_data = train_data.prefetch(tf.data.AUTOTUNE)


print("Preparing testing dataset...")

test_data = test_data.filter(keep_selected)
test_data = test_data.map(
    prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_data = test_data.batch(128)
test_data = test_data.prefetch(tf.data.AUTOTUNE)


# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        36, activation="softmax"
    )
])


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train model
print("Starting training...")

model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)


# Evaluate model
print("Evaluating model...")

loss, accuracy = model.evaluate(test_data)

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Save the combined model
model.save("handwritten_character_model.keras")

print("Combined model saved successfully!")
