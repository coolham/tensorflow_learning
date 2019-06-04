import sys
import os

import tensorflow as tf

print('TensorFlow:{}'.format(tf.__version__))
import numpy as np

print('NumPy:{}'.format(np.__version__))

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

current_path = os.getcwd()
base_dir = os.path.dirname(current_path)

if not base_dir in sys.path:
    sys.path.append(base_dir)

print(sys.path)
import datasetslib

mnist_home = os.path.join(datasetslib.datasets_root, 'mnist')
mnist = input_data.read_data_sets(mnist_home, one_hot=True)

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

num_outputs = 10
num_inputs = 784

tf.reset_default_graph()


# 多层感知机
def mlp(x, num_inputs, num_outputs, num_layers, num_neurons):
    w = []
    b = []
    for i in range(num_layers):
        # 权重
        w.append(tf.Variable(tf.random_normal(
            [num_inputs if i == 0 else num_neurons[i - 1], num_neurons[i]]),
            name='w_{0:04d}'.format(i)
        ))

        # 偏差
        b.append(tf.Variable(tf.random_normal([num_neurons[i]]), name='b_{0:04d}'.format(i)
                             ))

    w.append(tf.Variable(tf.random_normal(
        [num_neurons[num_layers - 1] if num_layers > 0 else num_inputs, num_outputs]),
        name='w_out'))
    b.append(tf.Variable(tf.random_normal([num_outputs]), name='b_out'))

    # x是输入层
    layer = x
    # 添加隐藏层
    for i in range(num_layers):
        layer = tf.nn.relu(tf.matmul(layer, w[i]) + b[i])
    # 添加输出层
    layer = tf.matmul(layer, w[num_layers]) + b[num_layers]

    return layer


def mnist_batch_func(batch_size=100):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    return [x_batch, y_batch]


def tensorflow_classification(n_epochs, n_batches, batch_size, batch_func, model, optimizer, loss, accuracy_function,
                              X_test, Y_test):
    accuracy_epochs = np.empty(shape=[n_epochs], dtype=np.float32)
    with tf.Session() as tfs:
        tfs.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in range(n_batches):
                X_batch, Y_batch = batch_func(batch_size)
                feed_dict = {x: X_batch, y: Y_batch}
                _, batch_loss = tfs.run([optimizer, loss], feed_dict)
                epoch_loss += batch_loss
            average_loss = epoch_loss / n_batches
            print('epoch {0:04d} loss = {1:.6f}'.format(epoch, average_loss))
            accuracy_epochs[epoch] = average_loss
        feed_dict = {x: X_test, y: Y_test}
        accuracy_score = tfs.run(accuracy_function, feed_dict=feed_dict)
        print('accuracy = {0:.8f}'.format(accuracy_score))

    plt.figure(figsize=(7, 4))
    plt.axis([0, n_epochs, np.min(accuracy_epochs), np.max(accuracy_epochs)])
    plt.plot(accuracy_epochs, label='Average Loss')
    plt.title('Loss over Iterations')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss Score')
    plt.legend()
    plt.show()

tf.reset_default_graph()

x = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='y')

num_layers = 2
num_neurons = []
for i in range(num_layers):
    num_neurons.append(256)

learning_rate = 0.01
n_epochs = 50
batch_size = 100
n_batches = int(mnist.train.num_examples / batch_size)

model = mlp(x=x,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_layers=num_layers,
            num_neurons=num_neurons)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

prediction_check = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy_function = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

tensorflow_classification(n_epochs=n_epochs,
                          n_batches=n_batches,
                          batch_size=batch_size,
                          batch_func=mnist_batch_func,
                          model=model,
                          optimizer=optimizer,
                          loss=loss,
                          accuracy_function=accuracy_function,
                          X_test=mnist.test.images,
                          Y_test=mnist.test.labels)


