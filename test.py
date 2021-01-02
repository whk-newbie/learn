from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

# #将索引解码为新闻文本
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value,key) for (value,key) in word_index.items()])
# decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])


#编码数据
import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


'''标签向量化，这里使用的是one-hot编码'''
# def to_one_hot(labels,dimension=64):
#     results = np.zeros((len(labels),dimension))
#     for i,label in enumerate(labels):
#         results[i,label] = 1.
#     return results
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)
'''keras 有内置方法'''
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#构建网络

from keras import models,layers

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]





history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val,y_val))



#画图
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
'''
 'b' 表示蓝色实线，'bo'表示蓝色圆点  
'''
plt.title('Training and Validation loss')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.legend()

plt.show()

plt.clf()  #清空图像

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')
plt.legend()

plt.show()

results = model.evaluate(x_test,one_hot_test_labels)
print(results)