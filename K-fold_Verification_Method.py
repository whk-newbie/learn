
import numpy as np

k = 4
num_val_samples = len(train_data)/k

# num_epochs = 100
# all_scores = []
#
# for i in range(k):
#     print('processing fold #',i)
#     val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
#     val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
#
#     partical_train_data = np.concatenate(
#         [train_data[i*num_val_samples],
#         train_data[(i+1)*num_val_samples:]],
#         axis=0
#     )
#     partical_train_targets = np.concatenate(
#             [train_targets[:i*num_val_samples],
#              train_targets[(i+1)*num_val_samples:]],
#             axis=0
#     )
#
#     model = build_model()
#     model.fit(partical_train_data,
#               partical_train_targets,
#               num_epochs=num_epochs,
#               batch_size=1,
#               verbose=0)
#
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)


#保存验证结果
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i*num_val_samples:(i+1)num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partical_train_data = np.concatenate(
            [train_data[i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0
    )

    partical_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history= model.fit(partical_train_data,
                partical_train_targets,
                num_epochs=num_epochs,
                batch_size=1,
                verbose=0)
    mae_histories = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_histories)


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]


import matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('EPOCHES')
plt.ylabel('VALIDDATION MAE')
plt.show