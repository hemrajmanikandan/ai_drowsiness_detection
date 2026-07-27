import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# =========================
# DATA GENERATOR (IMPORTANT)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.3,
    brightness_range=[0.5,1.5],
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    'data/yawn_dataset/train',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

val_data = train_datagen.flow_from_directory(
    'data/yawn_dataset/val',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# =========================
# MODEL (STRONG)
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

model.save("models/yawn_model.h5")

print("✅ Yawn model trained!")