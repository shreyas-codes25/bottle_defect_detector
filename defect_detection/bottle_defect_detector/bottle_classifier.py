import os
import cv2
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

# Paths
data_dir = "open in vs code , copy the path for Data folder and change '\' to '/'"
model_path = "final_bottle_classifier.h5"

# === TRAINING ===
def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=out)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=2, factor=0.2, verbose=1),
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks)

    # Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

    model.save(model_path)
    print("Model saved to:", model_path)

# === REAL-TIME CAMERA DETECTION WITH BOUNDING BOX ===
def predict_realtime():
    if not os.path.exists(model_path):
        print("Model not found! Please train the model first.")
        return

    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    class_labels = ['Defective', 'Proper']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dimensions and bounding box
        (h, w) = frame.shape[:2]
        box_size = int(min(h, w) * 0.6)
        start_x = (w - box_size) // 2
        start_y = (h - box_size) // 2
        end_x = start_x + box_size
        end_y = start_y + box_size

        # Draw bounding box
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)

        # Crop and predict
        cropped = frame[start_y:end_y, start_x:end_x]
        img = cv2.resize(cropped, (224, 224))
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = class_labels[int(prediction >= 0.5)]
        confidence = prediction if label == 'Proper' else 1 - prediction

        # Display result
        text = f'{label} ({confidence*100:.2f}%)'
        color = (0, 255, 0) if label == 'Proper' else (0, 0, 255)
        cv2.putText(frame, text, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Bottle Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === MAIN ENTRY ===
if __name__ == "__main__":
    # train_model()  # Uncomment to train the model
    predict_realtime()  # Real-time detection with bounding box
