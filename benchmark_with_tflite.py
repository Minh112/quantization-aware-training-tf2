import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image


IMG_PATH = "dataset/imagenet-100cls-60k/train/n01749939/n01749939_59.JPEG"

# Load the TFLite model and allocate tensors.
MODEL_PATH = "models/v3-large-minimalistic_224_1.0_uint8.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def get_img_uint8(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img


def get_tensor_test(num_multiply):
    tensor_test = []
    train_dir = "dataset/imagenet-100cls-60k/train"
    classes = os.listdir(train_dir)
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        for image_name in os.listdir(class_dir)[:num_multiply]:
            image_path = os.path.join(class_dir, image_name)
            image = get_img_uint8(image_path)
            tensor_test.append(image)
    tensor_test = np.array(tensor_test)

    return tensor_test


input_data = get_tensor_test(2)
# input_type = input_details[0]['dtype']
# print(input_type)
# if input_type == np.uint8:
#     input_scale, input_zero_point = input_details[0]['quantization']
#     print("Input scale:", input_scale)
#     print("Input zero point:", input_zero_point)

start = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
# i = np.argmax(output_data[0])
# print(i, output_data[0][i])
# end = time.time()
tensor_details = interpreter.get_tensor_details()
for i in range(len(tensor_details)):
    print(tensor_details[i])
print(interpreter.get_tensor(37))
end = time.time()
print(end - start)

