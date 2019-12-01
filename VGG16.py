from tensorflow import keras


model = keras.Sequential()
# block1
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(224, 224, 3),name="block1_conv1"))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv2"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="block1_MaxPool"))

# block2
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv1"))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv2"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="block2_MaxPool"))

# block3
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv1"))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv2"))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv3"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="block3_MaxPool"))

# block4
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv1"))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv2"))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv3"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="block4_MaxPool"))

# block5
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv1"))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv2"))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv3"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="block5_MaxPool"))

# block6
model.add(keras.layers.Flatten(name="block6_flatten"))
model.add(keras.layers.Dense(4096, activation="relu", name="block6_dense1"))
model.add(keras.layers.Dense(4096, activation="relu", name="block6_dense2"))
model.add(keras.layers.Dense(27, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

print(model.summary())

