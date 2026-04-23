import tensorflow as tf

# SETTINGS
img_size = (224, 224)
batch_size = 32

# LOAD DATASET
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:/AIML_Project/dataset/train",
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:/AIML_Project/dataset/test",
    image_size=img_size,
    batch_size=batch_size
)

# CLASS NAMES
class_names = train_ds.class_names
print("Classes:", class_names)

# CORRECT PREPROCESSING (IMPORTANT)
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

# PERFORMANCE BOOST
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# PRETRAINED MODEL
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# FIRST: FREEZE MODEL
base_model.trainable = False

# MODEL
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# EARLY STOPPING
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# TRAIN (PHASE 1)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop]
)

# PHASE 2: FINE-TUNING (IMPORTANT)
base_model.trainable = True

# Freeze lower layers again
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN AGAIN
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# SAVE MODEL
model.save("skin.h5")

print("Model saved successfully!")