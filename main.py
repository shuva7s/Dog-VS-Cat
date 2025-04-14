import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow using GPU:", tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'))

img_size = (150, 150)
batch_size = 64
input_shape = img_size + (3,)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    './dvc/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10
)

model.save('./dvc_tf2/cat_dog_model_full.h5')
