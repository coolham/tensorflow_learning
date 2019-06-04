import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets(os.path.join('.', 'mnist'), one_hot=True)

X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels


n_classes = 10    # 0~9位数
n_width = 28
n_height = 28
n_depth = 1
n_inputs = n_height * n_width * n_depth  #总像素

learning_rate = 0.001
n_epochs = 10
batch_size = 100
n_batches = int(mnist.train.num_examples/batch_size)

# 输入图像形状(n_samples, n_width, n_height, d_depth)
x = tf.placeholder(dtype=tf.float32, name='x', shape=[None, n_inputs])
# 输出标签
y = tf.placeholder(dtype=tf.float32, name='y', shape=[None, n_classes])

# 转换输入x为形状(n_samples, n_width, n_height, d_depth)
x_ = tf.reshape(x, shape=[-1, n_width, n_height, n_depth])

#  使用32个4x4大小的核定义第一个卷积层，从而生成32个特征图
# 首先，定义第一个卷积层的权重和偏差，使用正态分布初始化这些参数
layer1_w = tf.Variable(tf.random_normal(shape=[4, 4, n_depth, 32],stddev=0.1), name='l1_w')
layer1_b = tf.Variable(tf.random_normal([32]), name='l1_b')

# tf.nn.conv2d 定义卷积层
layer1_conv = tf.nn.relu(tf.nn.conv2d(x_, layer1_w, strides=[1, 1, 1, 1], padding='SAME') + layer1_b)

# tf.nn.max_pool 定义第一个池化层
# 第一个卷积层产生32个大小为28x28x1的特征图，然后池化成32x14x14x1
layer1_pool = tf.nn.max_pool(layer1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二个卷积层用上面的数据作为输入，并生成64个特征图

layer2_w = tf.Variable(tf.random_normal(shape=[4, 4, 32, 64], stddev=0.1), name='l2_w')
layer2_b = tf.Variable(tf.random_normal([64]), name='l2_b')

layer2_conv = tf.nn.relu(tf.nn.conv2d(layer1_pool, layer2_w, strides=[1,1,1,1,], padding='SAME') + layer2_b)

# 第二层卷积输出的大小为64x14x14x1， 池化之后为64x7x7x1
layer2_pool = tf.nn.max_pool(layer2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 在输入给全连接层（它有1024个神经元）之前需要将输出结果拉伸(flat)成大小为1024的向量

layer3_w = tf.Variable(tf.random_normal(shape=[64*7*7*1, 1024], stddev=0.1), name='l3_w')
layer3_b = tf.Variable(tf.random_normal([1024]), name='l3_b')
layer3_fc = tf.nn.relu(tf.matmul(tf.reshape(layer2_pool, [-1, 64*7*7*1]), layer3_w) + layer3_b)

# 全连接层的输出已一个线性输出层(它有10个输出)相连，
# 这一层没有使用softmax，因为损失函数会自动将softmax应用于输出
layer4_w = tf.Variable(tf.random_normal(shape=[1024, n_classes], stddev=0.1), name='l4_w')
layer4_b = tf.Variable(tf.random_normal([n_classes]), name='l4_b')
layer4_out = tf.matmul(layer3_fc, layer4_w) + layer4_b

# 创建第一个CNN模型，保存在变量model中
model = layer4_out

# 可用softmax_cross_entropy_with_logits定义损失函数
# 使用AdamOptimizer作为优化器
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as tfs:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in range(n_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed_dict = {x: batch_x, y: batch_y}
            batch_loss, _ = tfs.run([loss, optimizer], feed_dict=feed_dict)
            total_loss += batch_loss
        average_loss = total_loss / n_batches
        print('Epoch: {0:04d} loss = {1:0.6f}'.format(epoch, average_loss))
    print('Model Trained.')

    predictions_check = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
    feed_dict = {x: mnist.test.images, y: mnist.test.labels}
    print('Accuracy:', accuracy.eval(feed_dict=feed_dict))


