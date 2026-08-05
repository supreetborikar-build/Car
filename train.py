import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Dataset Path
# -----------------------------
DATA_DIR = "dataset"
CSV_FILE = os.path.join(DATA_DIR, "driving_log.csv")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(
    CSV_FILE,
    names=["center","left","right","steering","throttle","brake","speed"]
)

images = []
steerings = []

# -----------------------------
# Read Center Images Only
# -----------------------------
for _, row in data.iterrows():

    img_path = os.path.join(DATA_DIR, row["center"].strip())

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop sky & hood
    img = img[60:135, :, :]

    # Resize
    img = cv2.resize(img, (200, 66))

    images.append(img)
    steerings.append(float(row["steering"]))

X = np.array(images)
y = np.array(steerings)

print("Images :", X.shape)
print("Labels :", y.shape)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# NVIDIA CNN
# -----------------------------
model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)),

    Conv2D(24, (5,5), strides=(2,2), activation='elu'),
    Conv2D(36, (5,5), strides=(2,2), activation='elu'),
    Conv2D(48, (5,5), strides=(2,2), activation='elu'),
    Conv2D(64, (3,3), activation='elu'),
    Conv2D(64, (3,3), activation='elu'),

    Flatten(),

    Dense(100, activation='elu'),
    Dropout(0.5),
    Dense(50, activation='elu'),
    Dense(10, activation='elu'),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="mse"
)

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    shuffle=True
)

# -----------------------------
# Save Model
# -----------------------------
model.save("model.h5")

print("\n✅ Model saved as model.h5")
