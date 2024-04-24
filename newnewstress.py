import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

# Load MobileNet base model
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom dense layers
x = Flatten()(base_model.output)
x = Dense(units=2, activation='softmax')(x)
model = Model(base_model.input, x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define data generators
train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    directory="D:/facesData/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' class mode for multi-class classification
)

val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    directory="D:/facesData/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' class mode for multi-class classification
)

# Define early stopping and model checkpoint callbacks
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5, verbose=1, mode='auto')
mc = ModelCheckpoint(filepath="D:/best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks = [es, mc]

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=callbacks
)

# Load the best model
best_model = load_model("D:/best_model.keras")

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
