import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
image_path = 'C:/Users/User/OneDrive/Pictures/images22.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = preprocess_input(image_array)
image_array = tf.expand_dims(image_array, axis=0)

# Make a prediction
predictions = model.predict(image_array)
decoded_predictions = decode_predictions(predictions, top=5)

# Print the predicted objects and their confidence scores
for i, (class_id, label, confidence) in enumerate(decoded_predictions[0]):
    print(f"{i+1}. {label}: {confidence * 100:.2f}%")
