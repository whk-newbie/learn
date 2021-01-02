from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()



 #数据差异大时，进行数据标准化
mean = train_data.mean(axis=0) #求数据的特征平均值
std = train_data.std(axis=0) #求数据的标准差
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

from keras import models,layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model
'''
mse损失函数，即均方误差（MSE, mean squared error)
回归问题常用损失函数

监控指标mae 平均绝对误差（MAE,mean absolute error)
'''
#k折验证
import numpy as np
k = 4
num_val_samples = len(train_data)//k

num_epochs = 500

all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partical_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0
    )

    partical_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(partical_train_data,
                        partical_train_targets,
                        validation_data=(val_data,val_targets),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=0)
    mae_histories = history.history['val_mae']
    all_mae_histories.append(mae_histories)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('EPOCHES')
plt.ylabel('VALIDDATION MAE')
plt.show()


#绘制验证分数（删除前10个数据点)
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
           smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('EPOCHS')
plt.ylabel('VALIDATION MAE')
plt.show()

# model = build_model()
# model.fit(train_data,
#           train_targets,
#           epochs=8,
#           batch_size=16,
#           verbose=0)
# test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
# print(test_mae_score)