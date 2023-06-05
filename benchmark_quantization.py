import time

import tensorflow as tf
import keras
import numpy as np
import os
from PIL import Image
from keras.models import Model
from keras.layers import *
import json
from quantize_aware_training import get_img_array, benchmark_float_in_train
from models.mobilenetv3_minimalistic import MobileNet


def conv2d_dense_infer_quant(layer, data_quant, x):
    kernel = np.array(data_quant["kernel"])
    bias = np.array(data_quant["bias"])
    Z_k = data_quant["flt_off"]
    Z_i = data_quant["inp_off"]
    Z_o = data_quant["out_off"]
    M0 = data_quant["multiplier"]
    n = data_quant["shift"]
    x = x - Z_i
    if len(layer.get_weights()) == 2:
        layer.set_weights([kernel - Z_k, bias])
    else:
        layer.set_weights([kernel - Z_k])
    x = layer(x).numpy()
    x = np.floor(x * M0 / 2147483648 + 0.5, dtype=np.float32)
    x = np.floor(x / pow(2, -n) + 0.5, dtype=np.float32)
    x = x + Z_o
    return x


def add_infer_quant(data_quant, inp1, inp2):
    Z_i1 = data_quant["inp1_off"]
    Z_i2 = data_quant["inp2_off"]
    Z_o = data_quant["out_off"]
    M10 = data_quant["inp1_multiplier"]
    n1 = data_quant["inp1_shift"]
    M20 = data_quant["inp2_multiplier"]
    n2 = data_quant["inp2_shift"]
    Mo0 = data_quant["out_multiplier"]
    nofix = data_quant["out_shift"]
    nleft = data_quant["left_shift"]   # = 20
    no = nofix + nleft   # = 20 + nofix
    inp1 = np.floor((inp1 - Z_i1) * M10 / 2147483648 + 0.5, dtype=np.float32)
    inp1 = np.floor(inp1 / pow(2, -n1) + 0.5, dtype=np.float32)
    inp2 = np.floor((inp2 - Z_i2) * M20 / 2147483648 + 0.5, dtype=np.float32)
    inp2 = np.floor(inp2 / pow(2, -n2) + 0.5, dtype=np.float32)
    x = np.floor((inp1 + inp2) * Mo0 / 2147483648 + 0.5, dtype=np.float32)
    if no >= 0:
        x = (x * pow(2, no))
    else:
        x = np.floor(x / pow(2, -no) + 0.5, dtype=np.float32)
    x = x + Z_o
    # x = ReLU()(x).numpy()
    return x


def save_output_quant(output, path):
    '''
    Apply for test 1 image
    '''

    shape = output.shape
    if len(shape) == 4:
        num_value = shape[0] * shape[1] * shape[2] * shape[3]
    else:  # len(shape) = 2
        num_value = shape[0] * shape[1]
    output = output.reshape(num_value)
    text_output = ""
    num_full_line_output = num_value // 100
    for i in range(num_full_line_output):
        text_output += (str(list(output[100 * i: 100 * (i + 1)])).replace("[", "").replace("]", "") + ", \n")
    if num_value > num_full_line_output * 100:
        text_output += (str(list(output[100 * num_full_line_output:])).replace("[", "").replace("]", ""))
    with open(path, 'w') as f:
        f.write(text_output)


def benchmark_8bit_model(tensor_test, labels, h5_model, json_path):
    model = h5_model
    with open(json_path, "r") as f:
        model_quant_dict = json.load(f)

    output_dict = {}
    output_dict[model.layers[0].name] = tensor_test
    for i in range(len(model.layers)):
        name = model.layers[i].name
        # print(name)

        if "input" in name:
            info = model_quant_dict[name]
            Z, scale = info["out_off"], info["out_scale"]
            output = np.floor(tensor_test / scale + 0.5) + Z

        elif "conv2d" in name or "dense" in name:
            input_name = model.layers[i].input.name.split("/")[0]
            # print("--", input_name)
            input = output_dict[input_name].astype(float)
            data_quant = model_quant_dict[name]
            output = conv2d_dense_infer_quant(model.layers[i], data_quant, input)
            # output_dict[name] = output

        elif "add" in name and "padding" not in name:
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            # print("--", input_name1)
            # print("--", input_name2)
            input1 = output_dict[input_name1].astype(float)
            input2 = output_dict[input_name2].astype(float)
            data_quant = model_quant_dict[name]
            output = add_infer_quant(data_quant, input1, input2)
            for key in output_dict.keys():
                output_dict[key] = None
            # output_dict[name] = output

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            # print("--", input_name)
            input = output_dict[input_name].astype(float)
            output = model.layers[i](input).numpy()
            output = np.floor(output + 0.5)
            # output_dict[name] = output

        # print(np.min(output), np.max(output))
        output = ReLU(255)(output).numpy()
        output_dict[name] = output
        # print(output.shape)
        # print("------------")

    x = output_dict[model.layers[-1].name]
    test_out = [np.argmax(pred) for pred in x]
    score = 0
    for i in range(len(test_out)):
        if test_out[i] == labels[i]:
            score += 1
    print(score)
    print("acc quant: ", score/len(test_out))
    return score


def benchmark_float_model(tensor_test, labels, h5_model):
    model = h5_model
    Y_pred = model.predict(tensor_test)
    score = 0
    for i in range(len(Y_pred)):
        if np.argmax(Y_pred[i]) == labels[i]:
            score += 1
    acc = score / len(labels)
    print(acc)
    return acc


def benchmark_4bit_model(tensor_test, label, h5_path, json_path):
    model = keras.models.load_model(h5_path)
    with open(json_path, "r") as f:
        model_quant_dict = json.load(f)

    output_dict = {}
    output_dict[model.layers[0].name] = tensor_test
    for i in range(len(model.layers)):
        name = model.layers[i].name
        print(name)

        if "input" in name:
            info = model_quant_dict[name]
            Z, scale = info["out_off"], info["out_scale"]
            tensor_test = (tensor_test.astype(float) - 128) / 127.
            output = np.floor(tensor_test / scale + 0.5) + Z
            output_dict[name] = tensor_test

        elif "conv2d" in name or "dense" in name:
            input_name = model.layers[i].input.name.split("/")[0]
            print("--", input_name)
            input = output_dict[input_name].astype(float)
            data_quant = model_quant_dict[name]
            output = conv2d_dense_infer_quant(model.layers[i], data_quant, input)
            print(np.min(output), np.max(output))
            # output_dict[name] = output

        elif "add" in name and "padding" not in name:
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            print("--", input_name1)
            print("--", input_name2)
            input1 = output_dict[input_name1].astype(float)
            input2 = output_dict[input_name2].astype(float)
            data_quant = model_quant_dict[name]
            output = add_infer_quant(data_quant, input1, input2)
            print(np.min(output), np.max(output))
            for key in output_dict.keys():
                output_dict[key] = None
            # output_dict[name] = output

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            print("--", input_name)
            input = output_dict[input_name].astype(float)
            output = model.layers[i](input).numpy()
            output = np.floor(output + 0.5)
            print(np.min(output), np.max(output))
            # output_dict[name] = output

        output = ReLU(15)(output).numpy()
        output_dict[name] = output
        print(output.shape)
        print("***********")
        # txt_output = f"saved_models/txt_output/{name}_out.txt"
        # save_output_quant(output, txt_output)

    x = output_dict[model.layers[-1].name]
    test_out = [np.argmax(pred) for pred in x]
    score = 0
    for i in range(len(test_out)):
        if test_out[i] == label[i]:
            score += 1
    print(score)
    print("acc quant: ", score/len(test_out))


def benchmark_7bit_model(tensor_test, label, h5_path, json_path):
    model = keras.models.load_model(h5_path)
    with open(json_path, "r") as f:
        model_quant_dict = json.load(f)

    output_dict = {}
    output_dict[model.layers[0].name] = tensor_test
    for i in range(len(model.layers)):
        name = model.layers[i].name
        print(name)

        if "input" in name:
            info = model_quant_dict[name]
            Z, scale = info["out_off"], info["out_scale"]
            tensor_test = (tensor_test.astype(float) - 128) / 127.
            output = np.floor(tensor_test / scale + 0.5) + Z
            output_dict[name] = tensor_test

        elif "conv2d" in name or "dense" in name:
            input_name = model.layers[i].input.name.split("/")[0]
            print("--", input_name)
            input = output_dict[input_name].astype(float)
            data_quant = model_quant_dict[name]
            output = conv2d_dense_infer_quant(model.layers[i], data_quant, input)
            print(np.min(output), np.max(output))
            # output_dict[name] = output

        elif "add" in name and "padding" not in name:
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            print("--", input_name1)
            print("--", input_name2)
            input1 = output_dict[input_name1].astype(float)
            input2 = output_dict[input_name2].astype(float)
            data_quant = model_quant_dict[name]
            output = add_infer_quant(data_quant, input1, input2)
            print(np.min(output), np.max(output))
            for key in output_dict.keys():
                output_dict[key] = None
            # output_dict[name] = output

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            print("--", input_name)
            input = output_dict[input_name].astype(float)
            output = model.layers[i](input).numpy()
            output = np.floor(output + 0.5)
            print(np.min(output), np.max(output))
            # output_dict[name] = output

        output = ReLU(15)(output).numpy()
        output_dict[name] = output
        print(output.shape)
        print("***********")
        # txt_output = f"saved_models/txt_output/{name}_out.txt"
        # save_output_quant(output, txt_output)

    x = output_dict[model.layers[-1].name]
    test_out = [np.argmax(pred) for pred in x]
    score = 0
    for i in range(len(test_out)):
        if test_out[i] == label[i]:
            score += 1
    print(score)
    print("acc quant: ", score/len(test_out))


if __name__ == "__main__":
    # get model need to quantize
    h5_model = MobileNet(224)()
    h5_model.load_weights("models/3rd-train-float-w.h5")

    def benchmark_100samples(i):
        # get tensor quantize
        tensor_test = []
        labels = []
        val_dir = "dataset/imagenet-100cls-60k/val"
        classes = os.listdir(val_dir)
        label = 0

        for class_name in classes:
            class_dir = os.path.join(val_dir, class_name)
            for image_name in os.listdir(class_dir)[i:i+1]:
                image_path = os.path.join(class_dir, image_name)
                image = get_img_array(image_path)
                tensor_test.append(image)
                labels.append(label)
            label += 1
        tensor_test = np.array(tensor_test)
        labels = np.array(labels)

        # benchmark
        json_model = "quantization_models/3rd-train-float-uint8.json"
        # start = time.time()
        score = benchmark_8bit_model(tensor_test, labels, h5_model, json_model)
        # end = time.time()
        # print("time: ", end - start)
        return score

    scores = 0
    for i in range(120):
        score = benchmark_100samples(i)
        scores += score
        print("----", i, "----")
    print("********")
    print(scores)
    print("total acc: ", scores / 12000.)


    # # print("-----------")
    # # benchmark_float_model(tensor_test, labels, h5_model)

    # # benchmark all val
    # float_acc = benchmark_float_in_train(batch_val=120,
    #                                      model=h5_model,
    #                                      verbose=1)
    # print(float_acc)
