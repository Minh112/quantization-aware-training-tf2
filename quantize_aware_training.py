import os
import tensorflow as tf
import keras
import numpy as np
import json
from sklearn.utils import shuffle
from models.mobilenetv3_minimalistic import MobileNet
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tqdm import tqdm
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from keras.layers import *


DATA_DIR = "dataset/imagenet-100cls-60k"


def get_X_Y(img_dir):
    classes = os.listdir(img_dir)
    X = []
    Y = []
    for i in range(len(classes)):
        class_dir = os.path.join(img_dir, classes[i])
        images = os.listdir(class_dir)
        images = [os.path.join(class_dir, image) for image in images]
        labels = [i for image in images]
        X += images
        Y += labels
    return X, Y


def get_img_array(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.

    return img


class Dataset:
    def __init__(self, data, label):
        # the paths of images
        self.data = np.array(data)
        # the paths of segmentation images

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # read data
        image = get_img_array(self.data[i])
        label = to_categorical(self.label[i], num_classes=100)
        return image, label


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = size

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return self.size // self.batch_size


def train_generator(model, epochs, batch_size, save_path, pretrained=None):
    model.compile(optimizer=Adam(lr=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    if pretrained:
        model.load_weights(pretrained)

    train_dir = f"{DATA_DIR}/train"
    val_dir = f"{DATA_DIR}/val"
    train_processdata_func = ImageDataGenerator(rescale=1./255)
    train_data = train_processdata_func.flow_from_directory(directory=train_dir, target_size=(224, 224),
                                                            class_mode='categorical', batch_size=batch_size)
    val_data = train_processdata_func.flow_from_directory(directory=val_dir, target_size=(224, 224),
                                                          class_mode='categorical', batch_size=batch_size)

    # info = {
    #         "acc_max": 0.,
    #         "val_acc_max": 0.
    #         }
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch}:")
    #     train = model.fit_generator(generator=train_data, epochs=1, steps_per_epoch=200,
    #                                 validation_data=val_data, validation_steps=20,
    #                                 verbose=1)
    #     acc = train.history['acc'][0]
    #     val_acc = train.history['val_acc'][0]
    #     if val_acc > info['val_acc_max']:
    #         info['val_acc_max'] = val_acc
    #         model.save_weights("models/1st-train-best_val-w.h5")
    #         info['best_val'] = {
    #                             "val_acc": val_acc,
    #                             "acc": acc,
    #                             "epoch id": epoch
    #                             }
    #         if acc > info['acc_max']:
    #             model.save_weights("models/1st-train-best_val_train-w.h5")
    #             info['best_val_train'] = {
    #                                       "val_acc": val_acc,
    #                                       "acc": acc,
    #                                       "epoch id": epoch
    #                                       }
    #     if acc > info['acc_max']:
    #         info['acc_max'] = acc
    #
    # save_info_path = "models/1st-train-info.json"
    # with open(save_info_path, "w") as f:
    #     json.dump(info, f)

    my_callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=50),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=15),
                    tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                       save_weights_only=True,
                                                       monitor='val_acc',
                                                       mode='max',
                                                       save_freq="epoch",
                                                       save_best_only=True,
                                                       verbose=1)
                    ]
    steps_per_epoch = 48000 // batch_size
    validation_steps = 12000 // batch_size
    model.fit_generator(generator=train_data, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_data=val_data, validation_steps=validation_steps,
                        verbose=1, callbacks=my_callbacks)
    # model.save_weights(save_path)


def train_gen2(model, epochs, batch_size, save_path, pretrained=None):
    X_train_raw, Y_train_raw = get_X_Y(f"{DATA_DIR}/train")
    print(len(X_train_raw))
    X_val_raw, Y_val_raw = get_X_Y(f"{DATA_DIR}/val")
    print(len(X_val_raw))
    steps_per_epoch = len(X_train_raw) // batch_size
    validation_steps = len(X_val_raw) // batch_size

    train_dataset = Dataset(X_train_raw, Y_train_raw)
    val_dataset = Dataset(X_val_raw, Y_val_raw)
    train_loader = Dataloader(train_dataset, batch_size, len(train_dataset))
    val_loader = Dataloader(val_dataset, batch_size, len(val_dataset))

    model.compile(optimizer=Adam(lr=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    if pretrained:
        model.load_weights(pretrained)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=15),
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                           save_weights_only=True,
                                           monitor='val_acc',
                                           mode='max',
                                           save_freq="epoch",
                                           save_best_only=True,
                                           verbose=1)
    ]
    model.fit_generator(generator=train_loader, epochs=epochs,
                        validation_data=val_loader,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=1, callbacks=my_callbacks)


def get_tensor_quantize(num_multiply):
    tensor_quantize = []
    train_dir = "dataset/imagenet-100cls-60k/train"
    classes = os.listdir(train_dir)
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        for image_name in os.listdir(class_dir)[:num_multiply]:
            image_path = os.path.join(class_dir, image_name)
            image = get_img_array(image_path)
            tensor_quantize.append(image)
    tensor_quantize = np.array(tensor_quantize)

    return tensor_quantize


def get_all_tensor(model, tensor):
    output_dict = {}
    for i in range(len(model.layers)):
        name = model.layers[i].name

        if "input" in name:
            output = tensor

        elif "add" in name and "padding" not in name:
            inputs = model.layers[i].input
            input_name1 = inputs[0].name.split("/")[0]
            input_name2 = inputs[1].name.split("/")[0]
            input1 = output_dict[input_name1]
            input2 = output_dict[input_name2]
            output = input1 + input2

        else:
            input_name = model.layers[i].input.name.split("/")[0]
            input = output_dict[input_name]
            output = model.layers[i](input).numpy()

        output_dict[name] = output

    return output_dict


def quantize_8bit_model_in_train(model, tensor):
    all_outputs_dict = get_all_tensor(model, tensor)
    for layer in model.layers:
        name = layer.name
        # print(name)

        if "conv2d" in name or "dense" in name:
            # quantize kernel
            w = layer.get_weights()
            kernel = w[0]
            S_kernel = (np.max(kernel) - np.min(kernel)) / 255.
            kernel = np.floor(kernel / S_kernel + 0.5)
            kernel = kernel * S_kernel
            if len(w) == 1:
                layer.set_weights([kernel])
            else:
                # quantize bias
                input_name = layer.input.name.split("/")[0]
                input = all_outputs_dict[input_name]
                S_input = (np.max(input) - np.min(input)) / 255.
                S_bias = S_input * S_kernel
                bias = w[1]
                bias = np.floor(bias / S_bias + 0.5)
                bias = bias * S_bias
                layer.set_weights([kernel, bias])

    return model


def benchmark_float_in_train(batch_val, model, verbose=0):
    X_val_raw, Y_val_raw = get_X_Y(f"{DATA_DIR}/val")
    steps_benchmark = len(X_val_raw) // batch_val
    score = 0
    for step in range(steps_benchmark):
        X_val_batch = X_val_raw[batch_val * step: batch_val * (step + 1)]
        Y_val_batch = Y_val_raw[batch_val*step : batch_val*(step+1)]
        X_val, Y_val = [], []
        for i in range(batch_val):
            x_val = get_img_array(X_val_batch[i])
            if x_val.shape == (224, 224, 3):
                X_val.append(x_val)
                Y_val.append(Y_val_batch[i])
        X_val = np.array(X_val)
        Y_pred = model.predict(X_val, batch_size=batch_val, verbose=verbose)
        count = 0
        for i in range(len(Y_pred)):
            if np.argmax(Y_pred[i]) == Y_val[i]:
                count += 1
        score += count
    val_acc = score / len(X_val_raw)

    return val_acc


def train(model, batch_size, epochs, save_path, pretrained=None):
    model.compile(optimizer=Adam(lr=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    if pretrained:
        model.load_weights(pretrained)

    X_train_raw, Y_train_raw = get_X_Y(f"{DATA_DIR}/train")
    print(len(X_train_raw))
    X_val_raw, Y_val_raw = get_X_Y(f"{DATA_DIR}/val")
    print(len(X_val_raw))
    steps_per_epoch = len(X_train_raw) // batch_size
    print(steps_per_epoch)
    num_classes = len(os.listdir(f"{DATA_DIR}/train"))
    val_dataset = Dataset(X_val_raw, Y_val_raw)
    val_loader = Dataloader(val_dataset, batch_size, len(val_dataset))
    validation_steps = len(X_val_raw) // batch_size
    tensor_quantize = get_tensor_quantize(2)

    for epoch in range(epochs):
        X_train_raw, Y_train_raw = shuffle(X_train_raw, Y_train_raw)
        X_val_raw, Y_val_raw = shuffle(X_val_raw, Y_val_raw)
        print("Epoch {}:".format(epoch))
        pbar = tqdm(range(steps_per_epoch))
        # for step in pbar:
        #     X_batch = X_train_raw[batch_size*step : batch_size*(step+1)]
        #     Y_batch = Y_train_raw[batch_size*step : batch_size*(step+1)]
        #     X_val_batch = X_val_raw[batch_size_val*step : batch_size_val*(step+1)]
        #     Y_val_batch = Y_val_raw[batch_size_val*step : batch_size_val*(step+1)]
        #
        #     X_train, Y_train, X_val, Y_val = [], [], [], []
        #     for i in range(batch_size):
        #         x_train = get_img_array(X_batch[i])
        #         if x_train.shape == (224, 224, 3):
        #             X_train.append(x_train)
        #             Y_train.append(Y_batch[i])
        #     for i in range(batch_size_val):
        #         x_val = get_img_array(X_val_batch[i])
        #         if x_val.shape == (224, 224, 3):
        #             X_val.append(x_val)
        #             Y_val.append(Y_val_batch[i])
        #
        #     X_train = np.array(X_train)
        #     Y_train = np.array(Y_train)
        #     Y_train = to_categorical(Y_train, num_classes=num_classes)
        #     X_val = np.array(X_val)
        #     Y_val = np.array(Y_val)
        #     Y_val = to_categorical(Y_val, num_classes=num_classes)
        #
        #     train = model2.fit(X_train, Y_train, validation_data=(X_val, Y_val),
        #                       verbose=0,
        #                       batch_size=batch_size,
        #                       )
        #     # print(train.history.keys())
        #     loss = train.history["loss"]
        #     val_loss = train.history["val_loss"]
        #     acc = train.history["acc"]
        #     val_acc = train.history["val_acc"]
        #     model2 = quantize_8bit_model_in_train(model2)
        #     # val_acc_quant = benchmark(model, (X_val, Y_val))
        #     pbar.set_description(f"loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc}")
        #     # pbar.set_description(f"loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc} | val_acc_quant: {val_acc_quant}")
        for step in pbar:
            X_batch = X_train_raw[batch_size * step: batch_size * (step + 1)]
            Y_batch = Y_train_raw[batch_size*step : batch_size*(step+1)]
            train_dataset = Dataset(X_batch, Y_batch)
            train_loader = Dataloader(train_dataset, batch_size, len(train_dataset))
            train = model.fit_generator(generator=train_loader, epochs=1,
                                        steps_per_epoch=1,
                                        verbose=0,
                                        )
            loss = train.history["loss"]
            acc = train.history["acc"]
            model = quantize_8bit_model_in_train(model, tensor_quantize)
            pbar.set_description(f"loss: {loss} | acc: {acc}")

        val_acc = benchmark_float_in_train(batch_val=60, model=model)
        print(val_acc)
        model.save_weights(save_path)


if __name__ == "__main__":
    train(model=MobileNet(224)(),
          batch_size=120,
          epochs=10,
          save_path="models/4th-train-quant2-w.h5",
          pretrained="models/3rd-train-float-w.h5",
          )

    # train_generator(epochs=100,
    #                 batch_size=120,
    #                 save_path="models/3rd-train-float-w.h5",
    #                 pretrained="models/2nd-train-float-w.h5",
    #                 )

    # train_gen2(epochs=100,
    #            batch_size=120,
    #            save_path="models/3rd-train-float-w.h5",
    #            pretrained="models/2nd-train-float-w.h5",
    #            )

    # benchmark_float(120, "models/3rd-train-float-w.h5")

