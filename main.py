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

# from transformers import YolosImageProcessor, YolosForObjectDetection
# from PIL import Image, ImageDraw
# import torch
# import requests
#
# # url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.DcSXIZExW2M45haF40t9pAHaE8%26pid%3DApi&f=1&ipt=b1831aebe55145edcb3cfde593f789f030155ff8f093047c76bbd99f45cd4860&ipo=images"
# # image = Image.open(requests.get(url, stream=True).raw)
# image =Image.open('/home/ubuntu/Pictures/123.jpeg')
#
# model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
# image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
#
# inputs = image_processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# _ = model.save_pretrained("./model-dir")
# # model predicts bounding boxes and corresponding COCO classes
# logits = outputs.logits
# bboxes = outputs.pred_boxes
#
#
# # print results
# target_sizes = torch.tensor([image.size[::-1]])
# results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
#
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     draw = ImageDraw.Draw(image)
#     draw.rectangle(box)
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )
# image.show()
