
import os
import keras
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Reshape
from keras.optimizers import SGD

tf.reset_default_graph()
keras.backend.clear_session()


mnist = input_data.read_data_sets(os.path.join('.', 'mnist'), one_hot=True)

X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels


# 定义每个层的滤波器数量
n_filters = [32, 64]

learning_rate = 0.01
n_epochs = 10
batch_size = 100

n_classes = 10    # 0~9位数
n_width = 28
n_height = 28
n_depth = 1
n_inputs = n_height * n_width * n_depth  #总像素


model = Sequential()
model.add(Reshape(target_shape=(n_width, n_height, n_depth),
                  input_shape=(n_inputs, )
                  )
          )

# 使用4x4大小的滤波器
model.add(Conv2D(filters=n_filters[0],
                 kernel_size=4,
                 padding='SAME',
                 activation='relu'
                 )
          )

#添加区域大小为2x2， 且步长为2x2的池化层
model.add(MaxPool2D(pool_size=(2, 2),
                    strides=(2, 2)
                    )
          )

#添加第二个卷积和池化层
model.add(Conv2D(filters=n_filters[1],
                 kernel_size=4,
                 padding='SAME',
                 activation='relu'
                 )
          )
model.add(MaxPool2D(pool_size=(2, 2),
                    strides=(2, 2)
                    )
          )

#添加一个层做为第二层的输出，并添加1024个神经元的全连接层
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))

#将softmax激活函数添加到最后的输出层
model.add(Dense(units=n_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate),
              metrics=['accuracy']
              )
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          epochs=n_epochs
          )

score = model.evaluate(X_test, Y_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

