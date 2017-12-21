#coding=utf-8

import os
import numpy as np
import tensorflow as tf


def get_file(file_dir):
    cats =[]
    dogs =[]
    lable_cat = []
    lable_dog = []
    for file in os.listdir(file_dir):
        name = file.split(".")
        if name[0]=='cat':
            cats.append(file_dir+"/"+file)
            lable_cat.append(0)
        else:
            dogs.append(file_dir+"/"+file)
            lable_dog.append(1)
    print ('there are %d cats and %d dogs' % (len(cats),len(dogs)))
    image_list = np.hstack((cats,dogs))
    lable_list = np.hstack((lable_cat,lable_dog))

    temp = np.array([image_list,lable_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

def get_batch(image, label, image_w,image_h,batch_size ,capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_content,channels=3)

# picture resize to w * h
    image = tf.image.resize_image_with_crop_or_pad(image,image_w,image_h)

    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)

    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch,tf.float32)

    return image_batch,label_batch