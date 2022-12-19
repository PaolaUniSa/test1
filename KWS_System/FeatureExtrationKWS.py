import json
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sbn
data_path = "C:/Unisa/PhD/KWS/Audio_recognition/data.json"

#load dataset
with open(data_path, mode="r") as fp:
    data = json.load(fp)
    spectrograms_ds = np.array(data["MFFCs"])
    labels = np.array(data["mappings"])
    index_labels = np.array(data["labels"])
    path_files = np.array(data["files"])

signals = []

for wav_path in path_files:
    raw_audio = tf.io.read_file(wav_path)
    audio = tf.audio.decode_wav(raw_audio)
    signals.append(audio[0])


# plt.plot(signals[1])

# for i in signals:
#     i=i[..., np.newaxis]
arr = np.array(signals)



arr,index_labels= shuffle(arr,index_labels)
X_train,X_test,Y_train,Y_test=train_test_split(arr,index_labels,test_size=0.15)
X_train,X_validation,Y_train,Y_validation=train_test_split(X_train,Y_train,test_size=0.15)


#Convolutional autoencoder
encoder = keras.models.Sequential([
    keras.layers.Reshape([16000,1,1], input_shape=[16000,1]),
    keras.layers.Conv2D(32, kernel_size=(32,1), padding="same", activation="tanh"),
    keras.layers.Reshape([16000,32,1]),
    keras.layers.MaxPool2D(pool_size=(4,4)),
    keras.layers.Conv2D(16, kernel_size=(8,8), padding="same", activation="tanh"),
    keras.layers.Reshape([8000,64,1]),
    keras.layers.MaxPool2D(pool_size=(4,2)),
    keras.layers.Conv2D(2, kernel_size=(16,16), padding="same", activation="tanh"),
    keras.layers.MaxPool2D(pool_size=(4,4)),
    keras.layers.Reshape([2000, 4, 1]),
    keras.layers.Conv2D(2, kernel_size=(8,8), padding="same", activation="tanh"),
    keras.layers.Reshape([250, 64, 1]),
    keras.layers.MaxPool2D(pool_size=(5, 4)),
])

encoder.summary()

#
decoder = keras.models.Sequential([
    keras.layers.UpSampling2D(size=(5,1),input_shape=[50,16,1]),
    keras.layers.Reshape([250, 16, 1]),
    keras.layers.Conv2DTranspose(2, kernel_size=(16,16), padding="same",activation="tanh"),
    keras.layers.Reshape([500, 16,1]),
    keras.layers.Conv2DTranspose(2, kernel_size=(16,16), padding="same", activation="tanh"),
    keras.layers.Reshape([16000,1])
])
decoder.summary()

# decoder = keras.models.Sequential([keras.layers.Conv2DTranspose(2, kernel_size=(16,2), padding="same",activation="tanh"),
#         keras.layers.UpSampling2D(size=(5, 1),input_shape=[50,1, 2]),
#         keras.layers.Conv2DTranspose(2, kernel_size=(16,2), padding="same",activation="tanh"),
#         keras.layers.UpSampling2D(size=(4, 1)),
#         keras.layers.Conv2DTranspose(2, kernel_size=(32,2), padding="same", activation="tanh"),
#         keras.layers.UpSampling2D(size=(4, 1)),
#         #keras.layers.BatchNormalization(epsilon=1e-4,momentum=.1),
#         keras.layers.Conv2DTranspose(2, kernel_size=(32,2), padding="same", activation="tanh"),
#         keras.layers.UpSampling2D(size=(4, 1)),
#         keras.layers.Conv2DTranspose(1, kernel_size=(32,2), padding="same", activation="tanh"),
#         keras.layers.Reshape([16000,1])
# ])


#
model = keras.models.Sequential([encoder, decoder])
#
#
model.summary()


model.compile(loss='mse', optimizer='ADAM',metrics=['mae','mse'])

checkpoint_path = "C:/Unisa/PhD/KWS/models_feart/model_featureExtractionKWS.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose = 2, save_best_only=True)#,monitor='loss', mode='min')


# model.load_weights(checkpoint_path)
# ##################################################
batch_size=64
epochs=50

# history = model.fit(X_train, X_train, batch_size, epochs=epochs,
#                     callbacks=[checkpointer],
#                     validation_data=(X_validation, X_validation),
#                     # shuffle=True
#                     )
#
# # plt.plot(history.history["loss"], label="Training Loss")#validation_y
# # plt.plot(history.history["val_loss"], label="Validation Loss")
# # plt.legend()
#
model.load_weights(checkpoint_path)
# Evaluate the model on test set
score = model.evaluate(X_test, X_test, verbose=0)

# Print test accuracy
print('\n', 'Test mae:', score)


# #example
# i=1
# featureExtracted_example=encoder(tf.reshape(signals[i],[1,16000,1]))
# featureExtracted_example=featureExtracted_example.numpy()
# featureExtracted_example=featureExtracted_example.reshape(50,16)
#
# plt.figure(1)
# plt.plot(signals[i])
#
# plt.figure(2)
# plt.pcolormesh(featureExtracted_example)
# plt.close("all")

#freeze the base model
encoder.trainable= False

#KWS Classification
KWS_model = keras.models.Sequential()
model.add(keras.layers.BatchNormalization(input_shape=[50,16,1]))
# 1 Conv layer
KWS_model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", input_shape=[50,16,1], activation="tanh", kernel_regularizer=keras.regularizers.l2(0.0001)))
KWS_model.add(keras.layers.MaxPooling2D(pool_size= 2))
KWS_model.add(keras.layers.Dropout(0.3))
# 2 Conv layer
KWS_model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="tanh",kernel_regularizer=keras.regularizers.l2(0.0001)))
KWS_model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
KWS_model.add(keras.layers.Dropout(0.3))
# 3 Conv layer
KWS_model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="tanh",kernel_regularizer=keras.regularizers.l2(0.0001)))
KWS_model.add(keras.layers.Reshape([32,32, 1]))
KWS_model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
KWS_model.add(keras.layers.Dropout(0.3))
# Flatten and Dense layers
KWS_model.add(keras.layers.Flatten())
KWS_model.add(keras.layers.Dense(units=128, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.0001)))
KWS_model.add(keras.layers.Dropout(0.3))
KWS_model.add(keras.layers.Dense(units=len(labels), activation="softmax", kernel_regularizer=keras.regularizers.l2(0.0001)))


KWS_model.summary()

end_to_end_model = keras.models.Sequential([encoder, KWS_model])

end_to_end_model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])


checkpoint_path_end_to_end_model = "C:/Unisa/PhD/KWS/models_feart/end_to_end_model.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)
checkpointer_end_to_end_model = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_end_to_end_model,verbose = 2, save_best_only=True)#,monitor='loss', mode='min')

encoder.trainable= False
initial_encoder_weights_values = encoder.get_weights() ## check

history1 = end_to_end_model.fit(
    X_train,
    Y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_validation, Y_validation),
    callbacks=[checkpointer_end_to_end_model])


end_to_end_model.load_weights(checkpoint_path)

final_encoder_weights_values = encoder.get_weights() ## check

Y_pred=model.predict(X_test)
Y_pred=tf.argmax(Y_pred,axis=1)
cf_matrix=confusion_matrix(y_true=Y_test,y_pred=Y_pred,normalize='true')

for i in range(len(labels)):
        for j in range(len(labels)):
            cf_matrix[i][j]="%.2f" % cf_matrix[i][j]

ax=sbn.heatmap(data=cf_matrix,xticklabels=labels,yticklabels=labels,fmt='g',annot=True)
ax.set_xlabel('Predicted labels \n\n')
ax.set_ylabel('True labels ')
plt.show()