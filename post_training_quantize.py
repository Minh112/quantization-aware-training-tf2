import json
import os

import tensorflow as tf
import keras
import numpy as np
from keras.models import Model
import cv2
from PIL import Image
from quantize_aware_training import get_img_array
from models.mobilenetv3_minimalistic import MobileNet


def quant_uint8(tensor, activation=None):
    if activation == "relu" and np.min(tensor) < 0:
        a = 0
        b = np.max(tensor)
        S = (b - a) / 255.
        tensor = np.floor(tensor / S + 0.5)
        tensor = tensor.astype(int)
        Z = 0
    else:
        a = np.min(tensor)
        b = np.max(tensor)
        S = (b-a)/255.
        tensor = np.floor(tensor / S + 0.5)
        tensor = tensor.astype(int)
        Z = 0 - np.min(tensor)
        tensor = tensor + Z
    return Z, S, tensor


def quant_uint4(tensor, activation=None):
    if activation == "relu" and np.min(tensor) < 0:
        a = 0
        b = np.max(tensor)
        S = (b - a) / 15.
        tensor = np.floor(tensor / S + 0.5)
        tensor = tensor.astype(int)
        Z = 0
    else:
        a = np.min(tensor)
        b = np.max(tensor)
        S = (b-a)/15.
        tensor = np.floor(tensor / S + 0.5)
        tensor = tensor.astype(int)
        Z = 0 - np.min(tensor)
        tensor = tensor + Z
    return Z, S, tensor


def quant_uint7(tensor, activation=None):
    if activation == "relu" and np.min(tensor) < 0:
        a = 0
        b = np.max(tensor)
        S = (b - a) / 127.
        tensor = np.floor(tensor / S + 0.5)
        tensor = tensor.astype(int)
        Z = 0
    else:
        a = np.min(tensor)
        b = np.max(tensor)
        S = (b-a)/127.
        tensor = np.floor(tensor / S + 0.5)
        tensor = tensor.astype(int)
        Z = 0 - np.min(tensor)
        tensor = tensor + Z
    return Z, S, tensor


def quant_bias(tensor, S):
    tensor = np.floor(tensor / S + 0.5)
    tensor = tensor.astype(int)
    return tensor


def get_multiplier(M):
    M0, n = np.frexp(M)
    M0 = round(M0 * 2147483648)
    return M0, n


def quantize_8bit_all_layers(h5_model, tensor, json_path):
    model = h5_model
    all_layer_info = {}

    for i in range(len(model.layers)):
        layer_info = {}
        name = model.layers[i].name
        print(name)

        if "input" in name:
            Z = 0
            scale = 1 / 255.
            layer_info["out_off"] = Z
            layer_info["out_scale"] = scale

        elif "conv2d" in name or "dense" in name:
            subnet = Model(model.input, model.layers[i].output)
            outputs = subnet(tensor)
            output = outputs.numpy()
            input_name = model.layers[i].input.name.split("/")[0]
            input_info = all_layer_info[input_name]
            w = model.get_layer(name).get_weights()
            kernel = w[0]
            print(kernel.shape)
            Z_k, S_k, kernel_quant = quant_uint8(kernel)
            Z_o, S_o, _ = quant_uint8(output)
            Z_i, S_i = input_info["out_off"], input_info["out_scale"]
            if len(w) > 1:
                bias = w[1]
                S_bias = S_k * S_i
                bias_quant = quant_bias(bias, S_bias)
            else:
                bias_quant = np.zeros(shape=w[0].shape[-1])
            M = S_k * S_i / S_o
            M0, n = get_multiplier(M)
            # print(Z_k, S_k, Z_i, S_i, Z_o, S_o, S_bias, M0, n)
            layer_info["flt_off"] = int(Z_k)
            layer_info["inp_off"] = int(Z_i)
            layer_info["out_off"] = int(Z_o)
            layer_info["out_scale"] = S_o
            layer_info["multiplier"] = int(M0)
            layer_info["shift"] = int(n)
            layer_info["kernel"] = kernel_quant.tolist()
            layer_info["bias"] = bias_quant.tolist()
            if "conv2d" in name:
                layer_info["dilation"] = model.get_layer(name).dilation_rate

        elif "add" in name and "padding" not in name:
            subnet = Model(model.input, model.layers[i].output)
            outputs = subnet(tensor)
            output = outputs.numpy()
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            input_info1 = all_layer_info[input_name1]
            input_info2 = all_layer_info[input_name2]
            Z_o, S_o, _ = quant_uint8(output)
            Z_i1, S_i1 = input_info1["out_off"], input_info1["out_scale"]
            Z_i2, S_i2 = input_info2["out_off"], input_info2["out_scale"]
            if S_i2 < 2*S_i1:
                M1 = 1/2
                M2 = S_i2/(2*S_i1)
                Mo = 2*S_i1/S_o
            else:
                M1 = S_i1 / (2 * S_i2)
                M2 = 1/2
                Mo = 2*S_i2 / S_o
            M10, n1 = get_multiplier(M1)
            M20, n2 = get_multiplier(M2)
            Mo0, no = get_multiplier(Mo)
            layer_info["inp1_off"] = int(Z_i1)
            layer_info["inp2_off"] = int(Z_i2)
            layer_info["out_off"] = int(Z_o)
            layer_info["out_scale"] = S_o
            layer_info["inp1_multiplier"] = int(M10)
            layer_info["inp1_shift"] = int(n1)
            layer_info["inp2_multiplier"] = int(M20)
            layer_info["inp2_shift"] = int(n2)
            layer_info["out_multiplier"] = int(Mo0)
            layer_info["out_shift"] = int(no-20)
            layer_info["left_shift"] = 20

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            input_info = all_layer_info[input_name]
            Z_i, S_i = input_info["out_off"], input_info["out_scale"]
            layer_info["out_off"] = int(Z_i)
            layer_info["out_scale"] = S_i

        all_layer_info[name] = layer_info

    with open(json_path, "w") as f:
        json.dump(all_layer_info, f)

    return all_layer_info


def quantize_4bit_all_layers(h5_path, tensor, json_path):
    model = keras.models.load_model(h5_path)
    all_layer_info = {}

    for i in range(len(model.layers)):
        layer_info = {}
        name = model.layers[i].name
        print(name)

        if "input" in name:
            Z = 8
            scale = 1 / 7.
            layer_info["out_off"] = Z
            layer_info["out_scale"] = scale

        elif "conv2d" in name or "dense" in name:
            subnet = Model(model.input, model.layers[i].output)
            outputs = subnet(tensor)
            output = outputs.numpy()
            input_name = model.layers[i].input.name.split("/")[0]
            input_info = all_layer_info[input_name]
            w = model.get_layer(name).get_weights()
            kernel = w[0]
            print(kernel.shape)
            Z_k, S_k, kernel_quant = quant_uint4(kernel)
            Z_o, S_o, _ = quant_uint4(output)
            Z_i, S_i = input_info["out_off"], input_info["out_scale"]
            if len(w) > 1:
                bias = w[1]
                S_bias = S_k * S_i
                bias_quant = quant_bias(bias, S_bias)
            else:
                bias_quant = np.zeros(shape=w[0].shape[-1])
            M = S_k * S_i / S_o
            M0, n = get_multiplier(M)
            # print(Z_k, S_k, Z_i, S_i, Z_o, S_o, S_bias, M0, n)
            layer_info["flt_off"] = int(Z_k)
            layer_info["inp_off"] = int(Z_i)
            layer_info["out_off"] = int(Z_o)
            layer_info["out_scale"] = S_o
            layer_info["multiplier"] = int(M0)
            layer_info["shift"] = int(n)
            layer_info["kernel"] = kernel_quant.tolist()
            layer_info["bias"] = bias_quant.tolist()
            if "conv2d" in name:
                layer_info["dilation"] = model.get_layer(name).dilation_rate

        elif "add" in name and "padding" not in name:
            subnet = Model(model.input, model.layers[i].output)
            outputs = subnet(tensor)
            output = outputs.numpy()
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            input_info1 = all_layer_info[input_name1]
            input_info2 = all_layer_info[input_name2]
            Z_o, S_o, _ = quant_uint4(output)
            Z_i1, S_i1 = input_info1["out_off"], input_info1["out_scale"]
            Z_i2, S_i2 = input_info2["out_off"], input_info2["out_scale"]
            if S_i2 < 2 * S_i1:
                M1 = 1 / 2
                M2 = S_i2 / (2 * S_i1)
                Mo = 2 * S_i1 / S_o
            else:
                M1 = S_i1 / (2 * S_i2)
                M2 = 1 / 2
                Mo = 2 * S_i2 / S_o
            M10, n1 = get_multiplier(M1)
            M20, n2 = get_multiplier(M2)
            Mo0, no = get_multiplier(Mo)
            layer_info["inp1_off"] = int(Z_i1)
            layer_info["inp2_off"] = int(Z_i2)
            layer_info["out_off"] = int(Z_o)
            layer_info["out_scale"] = S_o
            layer_info["inp1_multiplier"] = int(M10)
            layer_info["inp1_shift"] = int(n1)
            layer_info["inp2_multiplier"] = int(M20)
            layer_info["inp2_shift"] = int(n2)
            layer_info["out_multiplier"] = int(Mo0)
            layer_info["out_shift"] = int(no - 20)
            layer_info["left_shift"] = 20

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            input_info = all_layer_info[input_name]
            Z_i, S_i = input_info["out_off"], input_info["out_scale"]
            layer_info["out_off"] = int(Z_i)
            layer_info["out_scale"] = S_i

        all_layer_info[name] = layer_info

    with open(json_path, "w") as f:
        json.dump(all_layer_info, f)

    return all_layer_info


def quantize_7bit_all_layers(h5_path, tensor, json_path):
    model = keras.models.load_model(h5_path)
    all_layer_info = {}

    for i in range(len(model.layers)):
        layer_info = {}
        name = model.layers[i].name
        print(name)

        if "input" in name:
            Z = 64
            scale = 1 / 63.
            layer_info["out_off"] = Z
            layer_info["out_scale"] = scale

        elif "conv2d" in name or "dense" in name:
            subnet = Model(model.input, model.layers[i].output)
            outputs = subnet(tensor)
            output = outputs.numpy()
            input_name = model.layers[i].input.name.split("/")[0]
            input_info = all_layer_info[input_name]
            w = model.get_layer(name).get_weights()
            kernel = w[0]
            print(kernel.shape)
            Z_k, S_k, kernel_quant = quant_uint7(kernel)
            Z_o, S_o, _ = quant_uint7(output)
            Z_i, S_i = input_info["out_off"], input_info["out_scale"]
            if len(w) > 1:
                bias = w[1]
                S_bias = S_k * S_i
                bias_quant = quant_bias(bias, S_bias)
            else:
                bias_quant = np.zeros(shape=w[0].shape[-1])
            M = S_k * S_i / S_o
            M0, n = get_multiplier(M)
            # print(Z_k, S_k, Z_i, S_i, Z_o, S_o, S_bias, M0, n)
            layer_info["flt_off"] = int(Z_k)
            layer_info["inp_off"] = int(Z_i)
            layer_info["out_off"] = int(Z_o)
            layer_info["out_scale"] = S_o
            layer_info["multiplier"] = int(M0)
            layer_info["shift"] = int(n)
            layer_info["kernel"] = kernel_quant.tolist()
            layer_info["bias"] = bias_quant.tolist()
            if "conv2d" in name:
                layer_info["dilation"] = model.get_layer(name).dilation_rate

        elif "add" in name and "padding" not in name:
            subnet = Model(model.input, model.layers[i].output)
            outputs = subnet(tensor)
            output = outputs.numpy()
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            input_info1 = all_layer_info[input_name1]
            input_info2 = all_layer_info[input_name2]
            Z_o, S_o, _ = quant_uint7(output)
            Z_i1, S_i1 = input_info1["out_off"], input_info1["out_scale"]
            Z_i2, S_i2 = input_info2["out_off"], input_info2["out_scale"]
            if S_i2 < 2*S_i1:
                M1 = 1/2
                M2 = S_i2/(2*S_i1)
                Mo = 2*S_i1/S_o
            else:
                M1 = S_i1 / (2 * S_i2)
                M2 = 1/2
                Mo = 2*S_i2 / S_o
            M10, n1 = get_multiplier(M1)
            M20, n2 = get_multiplier(M2)
            Mo0, no = get_multiplier(Mo)
            layer_info["inp1_off"] = int(Z_i1)
            layer_info["inp2_off"] = int(Z_i2)
            layer_info["out_off"] = int(Z_o)
            layer_info["out_scale"] = S_o
            layer_info["inp1_multiplier"] = int(M10)
            layer_info["inp1_shift"] = int(n1)
            layer_info["inp2_multiplier"] = int(M20)
            layer_info["inp2_shift"] = int(n2)
            layer_info["out_multiplier"] = int(Mo0)
            layer_info["out_shift"] = int(no-20)
            layer_info["left_shift"] = 20

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            input_info = all_layer_info[input_name]
            Z_i, S_i = input_info["out_off"], input_info["out_scale"]
            layer_info["out_off"] = int(Z_i)
            layer_info["out_scale"] = S_i

        all_layer_info[name] = layer_info

    with open(json_path, "w") as f:
        json.dump(all_layer_info, f)

    return all_layer_info


if __name__ == "__main__":
    # get tensor quantize
    tensor_quantize = []
    train_dir = "dataset/imagenet-100cls-60k/train"
    classes = os.listdir(train_dir)
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        for image_name in os.listdir(class_dir)[:2]:
            image_path = os.path.join(class_dir, image_name)
            image = get_img_array(image_path)
            tensor_quantize.append(image)
    tensor_quantize = np.array(tensor_quantize)
    print(np.max(tensor_quantize), np.min(tensor_quantize))
    print(tensor_quantize.shape)

    # get model need to quantize
    model = MobileNet(224)()
    model.load_weights("models/3rd-train-float-w.h5")
    save_json = "quantization_models/3rd-train-float-uint8.json"

    data = quantize_8bit_all_layers(model, tensor_quantize, save_json)
    print(data)

    # save_json_files("saved_models/mnist-res50-quant.json", "saved_models/json_files")
