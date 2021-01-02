from keras.datasets import imdb


# 加载数据集
(train_data,train_lables),(test_data,test_lables) = imdb.load_data(num_words=10000)
#仅保留前10000个最常出现单词

#将整数序列编码为二进制矩阵
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence  in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)   #数据向量化

y_train = np.asarray(train_lables).astype('float32')
y_test = np.asarray(test_lables).astype('float32')



#模型定义
from keras import models,layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#编译模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#配置优化器
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#使用自定义的损失和指标
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


#训练
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))


#绘图表示训练损失和验证损失
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.legend()

plt.show()

#绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')

plt.show()
'''
将此代码修改到前面的history中即可'''
# #防止过拟合从新建立模型
# model = models.Sequential()
# model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
# model.add(layers.Dense(16,activation='relu'))
# model.add(layers.Dense(1,activation='sigmod'))
#
# model.compile(optimizer='rmsprop',
#               loss='binaary_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=4,batch_size=512)
# results = model.evaluate(x_test,y_test)


model.predic(x_test)