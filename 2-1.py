from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()


#构建神经网络
from keras import models
from  keras import layers

#网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
#编译步骤
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
'''optimizer :优化器
    loss : 损失函数
    metric : 在训练过程中需要监控的指标
'''

#数据准备处理
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255


#准备标签

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss,test_acc = network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)

