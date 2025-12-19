import tensorflow as tf
import numpy as np
import pickle, os, cv2
import logging

logging.basicConfig(level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(os.listdir('gestures/'))

image_x, image_y = get_image_size()

class CNNModel(tf.keras.Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (2, 2), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((5, 5), strides=5)
        self.conv3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(get_num_of_classes(), activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

def main(argv):
    # Load data
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    # Reshape & normalize
    train_images = np.expand_dims(train_images, axis=-1).astype(np.float32) / 255.0
    test_images = np.expand_dims(test_images, axis=-1).astype(np.float32) / 255.0

    # Create model
    model = CNNModel()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        batch_size=500,
        validation_data=(test_images, test_labels)
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

    # Save model
    model.save("cnn_model_keras2.keras")
    print("Model saved as cnn_model_keras2.h5")

if __name__ == "__main__":
    import sys
    main(sys.argv)
