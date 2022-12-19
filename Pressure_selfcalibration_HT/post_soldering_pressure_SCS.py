# @authors: Paola Vitolo
# @Date: 01/07/2022
# @Brief: stesso modello dell'HT per il posto soldering


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import qkeras
from qkeras.autoqkeras import *
from qkeras import *


# ##################################################
# Ensure reprodcibility of random operations:
np.random.seed(123)

rng = tf.random.Generator.from_seed(123)
tf.random.set_global_generator(rng)
#####################################################################################
# Dataset
x_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/x_acc_in_ore_80devices.txt", delimiter=',') # shape=80x250 time (0->250)
y_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/y_acc_in_ore_80devices.txt", delimiter=',') # shape=80x250 accuracy of the 80 devices in post-soldering

min=np.min(y_dataset)
max=np.max(y_dataset)

y_dataset=(y_dataset-min)/(max-min)

# x_dataset=x_dataset/250

# 24 bits -> 2^24=16,777,216 values -> [-2^23 ; 2^23-1]= [-8388608 , 8388607]
y_dataset=(y_dataset*(pow(2,24)-1)) - pow(2,23)
y_dataset=y_dataset/(pow(2,12))

# x_dataset=(x_dataset*(pow(2,24)-1)) - pow(2,23)


x=x_dataset
y=y_dataset

input_shape=2  # (tempo, accuracy iniziale)
output_shape=1
# x=x[0:600]
# y=y[0:600]
# y=(y-np.min(y))/(np.max(y)-np.min(y))

init_vals=y[:,0]
init_vals_rep=init_vals.repeat(250,axis=0) # Acc[0]

x=x.reshape(80*250)
y=y.reshape(80*250)

x_values=np.array([x,init_vals_rep]).transpose() # [time, Acc[0]]

#####################################################################################
# Model
neuron_number1=9
neuron_number2=15
neuron_number3=9

# model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=[input_shape]),
#     keras.layers.Dense( neuron_number1,
#                         ),
#     keras.layers.Dense(neuron_number2,
#                         activation="sigmoid"
#                        ),
#     keras.layers.Dense(neuron_number3,
#                         # activation="relu"
#                        ),
#     keras.layers.Dense(output_shape,
#                        ),
# ])

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[input_shape]),

    QActivation("quantized_bits(bits=24, integer=12)"),
    # QActivation("quantized_bits(bits=24, integer=24, symmetric=1)"),
    # QActivation("quantized_bits(bits=24, integer=1)"),

    qkeras.QDense(neuron_number1, kernel_quantizer="quantized_bits(bits=24)", use_bias=False),

    QActivation("quantized_bits(bits=24)"),

    qkeras.QDense(neuron_number2, kernel_quantizer="quantized_bits(24)", use_bias=False),

    # QActivation("quantized_tanh(24)"),

    tf.keras.layers.Activation("sigmoid"),
    #
    QActivation("quantized_bits(bits=24)"),

    qkeras.QDense(neuron_number3, kernel_quantizer="quantized_bits(24)", use_bias=False),

    # QActivation("quantized_bits(bits=24)"),

    tf.keras.layers.Activation("relu"),

    QActivation("quantized_bits(bits=24)"),

    qkeras.QDense(output_shape, kernel_quantizer="quantized_bits(24)", use_bias=False),

    # QActivation("quantized_bits(bits=24)"),
])


model.summary()

qkeras.print_qstats(model)


model.compile(loss='mse', optimizer='Adam',metrics=['mae','mse'])
    # (loss='MeanAbsoluteError',
    #          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),#'adam',
    #          metrics=['MeanAbsoluteError'])




checkpoint_path = "C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/post_soldering_quantizedModel_vers1_test.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose = 2, save_best_only=True)#,monitor='loss', mode='min')
#
#
# # ##################################################
# # batch_size=256
epochs=50
# x=x.transpose()
# y=y.transpose()

history = model.fit(x=x_values, y=y, epochs=epochs,
                    # batch_size=30,
                    callbacks=[checkpointer],
                    validation_data=(x_values, y),
                    # shuffle=True
                    )

# plt.plot(history.history["loss"], label="Training Loss")validation_y
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()

model.load_weights(checkpoint_path)

# Evaluate the model on test set
score = model.evaluate(x_values,y, verbose=0)

# Print test accuracy
print('\n', 'Test mae:', score)

# pred=model.predict(x[0,:].reshape(1,input_shape_size))
# plt.close('all')
#
fig, axs = plt.subplots(6)
for i in range(6):
    pred=model.predict(np.array([x_dataset[i],y_dataset[i,0].repeat(250,axis=0)]).transpose())

    axs[i].plot(pred.reshape(250))
    axs[i].plot(y_dataset[i])


# plt.close("all")
#
#
x_dataset_sim = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/x_dataset_30sim_1h.txt", delimiter=',')
y_dataset_sim = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/y_dataset_30sim_1h.txt", delimiter=',')


j=1
accs=x_dataset_sim[j,1]-y_dataset_sim[j,1]
accs=(accs-min)/(max - min)
accs=accs.repeat(250,axis=0)

pred_sim=model.predict(np.array([x_dataset[0],accs]).transpose())

pred_sim=pred_sim*(max-min)+min

valore_corretto=x_dataset_sim[j].reshape(243)-pred_sim.reshape(250)[0:243]/100

plt.figure(2)
plt.plot(valore_corretto, 'g')
plt.plot(x_dataset_sim[j].reshape(243), 'r')
plt.plot(y_dataset_sim[j].reshape(243),'b')


print('\n', 'Mse(true values,drifted values):', mean_squared_error(y_dataset_sim[j].reshape(243), x_dataset_sim[j].reshape(243)))
print('\n', 'Mse(true values,corrected values):', mean_squared_error(y_dataset_sim[j].reshape(243), valore_corretto))