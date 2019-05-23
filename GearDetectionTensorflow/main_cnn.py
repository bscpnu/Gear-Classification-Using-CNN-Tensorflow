import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import prepro

from sklearn.metrics import confusion_matrix
from datetime import timedelta
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

## author: Imam Mustafa Kamal
## email: imamkamal52@gmail.com

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def print_progress(epoch, feed_dict_train):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)

    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}"
    print(msg.format(epoch + 1, acc))

def optimize(train_batch_size, num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        #x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        #x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        #feed_dict_validate = {x: x_valid_batch,
        #                      y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i % int(data.train.num_examples / batch_size) == 0:
            #val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))

            print_progress(epoch, feed_dict_train)

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time

    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

def summary_result(data_act, data_pred):

    cm = confusion_matrix(data_act, data_pred)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Defect', 'Non-Defect']
    plt.title('Confusion Matrix of Defect and Non-defect Classification')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()

    # calculate AUC
    auc = roc_auc_score(data_act, data_pred)
    print('AUC score = %.3f' % auc)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(data_act, data_pred)

    # plot no skill
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

    acc = accuracy_score(data_act, data_pred)
    print("Accuracy score = ", acc)

def write_predictions(data):
    ims = data.test.images.reshape(data.test.images.shape[0], img_size_flat)
    preds = session.run(y_pred, feed_dict={x: ims})

    data_id_test = pd.DataFrame(data.test.ids, columns=['image name'])
    data_labels_test = pd.DataFrame(data.test.labels, columns=['actual', 'actual_label'])
    data_class_predict = pd.DataFrame(preds, columns=['defect', 'non-defect'])

    data_act = np.where(data_labels_test['actual_label'] == 1, 'non-defect', 'defect')
    data_pred = np.where(data_class_predict['non-defect']>= 0.5, 'non-defect', 'defect')

    df_data_act = pd.DataFrame(data_act, columns=['actual'])
    df_data_pred = pd.DataFrame(data_pred, columns=['predicted'])

    result_test = pd.concat([data_id_test, df_data_act, df_data_pred], axis=1)

    df_result_test = pd.DataFrame(result_test)
    df_result_test.to_csv("result_test.csv")

    data_labels_pred = np.where(data_class_predict['non-defect'] >= 0.5, 1, 0)

    summary_result(data_labels_test['actual_label'], data_labels_pred)

if __name__=='__main__':
    prepro.crop_img("defect")
    prepro.crop_img("non_defect")
    # Convolutional Layer 1.
    filter_size1 = 3
    num_filters1 = 32

    # Convolutional Layer 2.
    filter_size2 = 3
    num_filters2 = 32

    # Convolutional Layer 3.
    filter_size3 = 3
    num_filters3 = 64

    # Fully-connected layer.
    fc_size = 128  # Number of neurons in fully-connected layer.

    # Number of color channels for the images: 1 channel for gray-scale.
    num_channels = 3

    # image dimensions (only squares for now)
    img_size = 256

    # Size of image when flattened to a single dimension
    img_size_flat = img_size * img_size * num_channels

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # class info
    classes = ['defect', 'non_defect']
    num_classes = len(classes)

    # batch size
    batch_size = 16

    # validation split
    validation_size = .16

    # how long to wait after validation loss stops improving before terminating training
    early_stopping = None  # use None if you don't want to implement early stoping

    data_path = 'prepro_data/'
    checkpoint_dir = "models/"

    data = dataset.read_data_sets(data_path, img_size, classes, portion_size=0.25)

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Testing-set:\t\t{}".format(len(data.test.labels)))


    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    y_true_cls = tf.argmax(y_true, dimension=1)

    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       use_pooling=True)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       use_pooling=True)

    layer_conv3, weights_conv3 = \
        new_conv_layer(input=layer_conv2,
                       num_input_channels=num_filters2,
                       filter_size=filter_size3,
                       num_filters=num_filters3,
                       use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv3)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)  # the class number is the index of largest element

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()

    session.run(tf.initialize_all_variables())

    train_batch_size = batch_size

    total_iterations = 0
    optimize(train_batch_size, num_iterations=100)  # We performed 1000 iterations above.

    write_predictions(data)
