import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('cat_dog_model.h5')

# List of tuples: (image_path, actual_label)
# actual_label: 0 = cat, 1 = dog
image_samples = [
    (r'E:\dogs-vs-cats\test1\test1\999.jpg', 0),    # cat
    (r'E:\dogs-vs-cats\test1\test1\9982.jpg', 1),   # dog
    (r'E:\dogs-vs-cats\test1\test1\9850.jpg', 1),   # dog
    (r'E:\dogs-vs-cats\test1\test1\9863.jpg', 0),   # cat
    (r'E:\dogs-vs-cats\test1\test1\9811.jpg', 0),   # cat
    (r'E:\dogs-vs-cats\test1\test1\9797.jpg', 1),   # dog
    (r'E:\dogs-vs-cats\test1\test1\9656.jpg', 0),   # cat
    (r'E:\dogs-vs-cats\test1\test1\9611.jpg', 0),   # cat
    (r'E:\dogs-vs-cats\test1\test1\9537.jpg', 1),   # dog
]

correct = 0

for img_path, actual in image_samples:
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)[0][0]
        predicted_label = 1 if prediction > 0.5 else 0
        label_str = "Dog 🐶" if predicted_label == 1 else "Cat 🐱"
        actual_str = "Dog 🐶" if actual == 1 else "Cat 🐱"

        # Accuracy tracking
        if predicted_label == actual:
            correct += 1
            result = "✅ Correct"
        else:
            result = "❌ Wrong"

        print(f"{img_path}: Predicted: {label_str} ({prediction:.2f}) | Actual: {actual_str} → {result}")

    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

print(f"\nAccuracy on test set: {correct}/{len(image_samples)} correct")
