#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head

import shutil


def _main(session, args_model_path, args_anchors_path, args_classes_path, args_test_path, args_output_path):
    model_path = args_model_path
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = args_anchors_path
    classes_path = args_classes_path
    test_path = args_test_path
    output_path = args_output_path

    args_score_threshold = .3
    args_iou_threshold = .5


    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    # sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    sess = session

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args_score_threshold,
        iou_threshold=args_iou_threshold)

    for image_file in os.listdir(test_path):
        # try:
        #     image_type = imghdr.what(os.path.join(test_path, image_file))
        #     if not image_type:
        #         continue
        # except IsADirectoryError:
        #     continue

        image = Image.open(os.path.join(test_path, image_file))
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        max_score = 0

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # del draw

            if predicted_class == 'dog':
                if score > max_score:
                    if max_score > 0:
                        print('-' * 10)
                    border = 10
                    max_score = score
                    crop_box = left - border, top - border, right + border, bottom + border
                    cropped_img = image.crop(crop_box)
                    cropped_img.save(os.path.join(output_path, image_file), quality=90)
            else:
                shutil.copyfile(os.path.join(test_path, image_file), os.path.join(output_path, image_file))

        # image.save(os.path.join(output_path, image_file), quality=90)

def _main_input():
    model_path = 'model_data/yolo.h5'
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/pascal_classes.txt'
    # model_path = args_model_path
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    # anchors_path = args_anchors_path
    # classes_path = args_classes_path
    # test_path = args_test_path
    # output_path = args_output_path
    intput_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input'
    data_folders = ['data_train', 'data_val', 'data_test']


    args_score_threshold = .3
    args_iou_threshold = .5

    count_max_dog = 0
    count_no_dog = 0
    count_no_object = 0


    # if not os.path.exists(output_path):
    #     print('Creating output path {}'.format(output_path))
    #     os.mkdir(output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    # sess = session

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args_score_threshold,
        iou_threshold=args_iou_threshold)

    
    for data_folder_name in data_folders:
        data_folder = os.path.join(intput_path, data_folder_name)
        output_folder = os.path.join(intput_path, 'yolo_' + data_folder_name)
        if not os.path.exists(output_folder):
            print('Create folders: %s' % output_folder)
            os.makedirs(output_folder)
        else:
            print('Folder exists: %s' % output_folder)

        for class_folder_name in os.listdir(data_folder):
            test_path = os.path.join(data_folder, class_folder_name)
            output_path = os.path.join(output_folder, class_folder_name)
            if not os.path.exists(output_path):
                print('Create folders: %s' % output_path)
                os.makedirs(output_path)
            else:
                print('Folder exists: %s' % output_path)

            for image_file in os.listdir(test_path):
                # try:
                #     image_type = imghdr.what(os.path.join(test_path, image_file))
                #     if not image_type:
                #         continue
                # except IsADirectoryError:
                #     continue

                image = Image.open(os.path.join(test_path, image_file))
                if is_fixed_size:  # TODO: When resizing we can use minibatch input.
                    resized_image = image.resize(
                        tuple(reversed(model_image_size)), Image.BICUBIC)
                    image_data = np.array(resized_image, dtype='float32')
                else:
                    # Due to skip connection + max pooling in YOLO_v2, inputs must have
                    # width and height as multiples of 32.
                    new_image_size = (image.width - (image.width % 32),
                                    image.height - (image.height % 32))
                    resized_image = image.resize(new_image_size, Image.BICUBIC)
                    image_data = np.array(resized_image, dtype='float32')
                    print(image_data.shape)

                image_data /= 255.
                image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

                try:
                    out_boxes, out_scores, out_classes = sess.run(
                        [boxes, scores, classes],
                        feed_dict={
                            yolo_model.input: image_data,
                            input_image_shape: [image.size[1], image.size[0]],
                            K.learning_phase(): 0
                        })
                except Exception as ex:
                    print('Err: %s' % image_file)
                    print(ex)
                    shutil.copyfile(os.path.join(test_path, image_file), os.path.join(output_path, image_file))
                    continue
                    

                # print('Found {} boxes for {}'.format(len(out_boxes), image_file))

                font = ImageFont.truetype(
                    font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 300
                
                max_score = 0
                
                if len(out_classes) > 0:
                    for i, c in reversed(list(enumerate(out_classes))):
                        predicted_class = class_names[c]
                        box = out_boxes[i]
                        score = out_scores[i]

                        label = '{} {:.2f}'.format(predicted_class, score)

                        draw = ImageDraw.Draw(image)
                        label_size = draw.textsize(label, font)

                        top, left, bottom, right = box
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                        # print(label, (left, top), (right, bottom))

                        if top - label_size[1] >= 0:
                            text_origin = np.array([left, top - label_size[1]])
                        else:
                            text_origin = np.array([left, top + 1])

                        # # My kingdom for a good redistributable image drawing library.
                        # for i in range(thickness):
                        #     draw.rectangle(
                        #         [left + i, top + i, right - i, bottom - i],
                        #         outline=colors[c])
                        # draw.rectangle(
                        #     [tuple(text_origin), tuple(text_origin + label_size)],
                        #     fill=colors[c])
                        # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                        # del draw

                        if predicted_class == 'dog':
                            if score > max_score:
                                if max_score > 0:
                                    print('+' * 10)
                                    count_max_dog += 1
                                border = 10
                                max_score = score
                                crop_box = left - border, top - border, right + border, bottom + border
                                cropped_img = image.crop(crop_box)
                                cropped_img.save(os.path.join(output_path, image_file), quality=90)
                        else:
                            count_no_dog += 1
                            print('-' * 10)
                            shutil.copyfile(os.path.join(test_path, image_file), os.path.join(output_path, image_file))
                else:
                    count_no_object += 1
                    print('*' * 10)
                    shutil.copyfile(os.path.join(test_path, image_file), os.path.join(output_path, image_file))
                
    print('%s %s %s' %(count_max_dog, count_no_dog, count_no_object))
                # image.save(os.path.join(output_path, image_file), quality=90)



if __name__ == '__main__':
    # sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    
    # 测试YOLO自带的图片
    model_path = 'model_data/yolo.h5'
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/pascal_classes.txt'
    # test_path = 'images'
    # output_path = 'images/out'
    # _main(model_path, anchors_path, classes_path, test_path, output_path)

    # 处理inputdata
    _main_input()

    # # 处理data_train
    # test_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input/data_train'
    # output_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input/yolo_data_train'
    # for folder_name in os.listdir(test_path):
    #     in_path = os.path.join(test_path, folder_name)
    #     out_path = os.path.join(output_path, folder_name)
    #     if not os.path.exists(out_path):
    #         print('Create folder: %s' % out_path)
    #         os.makedirs(out_path)
    #     else:
    #         print('Folder exists: %s' % out_path)

    #     # _main(sess, model_path, anchors_path, classes_path, in_path, out_path)

    # # 处理data_val
    # test_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input/data_val'
    # output_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input/yolo_data_val'
    # for folder_name in os.listdir(test_path):
    #     in_path = os.path.join(test_path, folder_name)
    #     out_path = os.path.join(output_path, folder_name)
    #     if not os.path.exists(out_path):
    #         print('Create folder: %s' % out_path)
    #         os.makedirs(out_path)
    #     else:
    #         print('Folder exists: %s' % out_path)

    #     # _main(sess, model_path, anchors_path, classes_path, in_path, out_path)

    # # 处理data_test
    # test_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input/data_test'
    # output_path = 'D:/Udacity/MachineLearning(Advanced)/p6_graduation_project/input/yolo_data_test'
    # for folder_name in os.listdir(test_path):
    #     in_path = os.path.join(test_path, folder_name)
    #     out_path = os.path.join(output_path, folder_name)
    #     if not os.path.exists(out_path):
    #         print('Create folder: %s' % out_path)
    #         os.makedirs(out_path)
    #     else:
    #         print('Folder exists: %s' % out_path)

    #     # _main(sess, model_path, anchors_path, classes_path, in_path, out_path)

    # sess.close()
