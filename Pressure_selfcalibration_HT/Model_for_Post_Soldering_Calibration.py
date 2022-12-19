import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# import qkeras
# from qkeras.autoqkeras import *
# from qkeras import *

# ##################################################
# Ensure reproducibility of random operations:
np.random.seed(123)

rng = tf.random.Generator.from_seed(123)
tf.random.set_global_generator(rng)

#####################################################################################
# Dataset for training
root_path="C:/Users/paola/OneDrive/Desktop/Self_Calibration System_Pressure_Sensors/Post Soldering/"
x_dataset = np.loadtxt(root_path+"Data/TrainingData/time_fitted_post_soldering_accuracy_80devices.txt", delimiter=',') # shape=80x250 time (0->249)
y_dataset = np.loadtxt(root_path+"Data/TrainingData/accuracy_fitted_post_soldering_accuracy_80devices.txt", delimiter=',') # shape=80x250 accuracy of the 80 devices in post-soldering

# Normalization between 0 and 1
## 24 bits -> 2^24=16,777,216 values -> [-2^23 ; 2^23-1]/sensitivity= [-8388608 , 8388607] /[2^12/hPa] --> min=-2^23/2^12 and max=(2^23-1)/2^12=2047.99
# min=-2048*2   # there is the multiplication of 2 because the accuracy is the difference between 2 signals coded with 24 bits
# max=2047.99*2

min=260-1260
max=1260+260

x=x_dataset
y=y_dataset

input_shape=2  # (tempo, accuracy iniziale)
output_shape=1

init_accuracy=y[:,1]-y[:,0]  # initial accuracy
init_vals_rep=init_accuracy.repeat(250,axis=0)

init_vals_rep=(init_vals_rep-min)/(max-min)
y=(y-min)/(max-min)



x=x.flatten()
y=y.flatten()

x=np.array([x,init_vals_rep]).transpose() # [time, Acc[0]]

# Split in training, validation, and test dataset
x_training_dataset= x[0:int(0.8*x.shape[0])]
x_validation_dataset= x[int(0.8*x.shape[0]):int(0.9*x.shape[0])]
x_test_dataset=x[int(0.9*x.shape[0]):x.shape[0]]

y_training_dataset= y[0:int(0.8*x.shape[0])]
y_validation_dataset=y[int(0.8*x.shape[0]):int(0.9*x.shape[0])]
y_test_dataset=y[int(0.9*x.shape[0]):x.shape[0]]

#####################################################################################
# Model
neuron_number1=9
neuron_number2=15
neuron_number3=9


model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[input_shape]),

    keras.layers.Dense(neuron_number1, use_bias=False),

    keras.layers.Reshape((9,1)),

    keras.layers.Conv1D(1, 3, use_bias=False),

    tf.keras.layers.Activation("sigmoid"),

    keras.layers.Conv1D(1, 3, use_bias=False),

    # tf.keras.layers.Activation("relu"),

    keras.layers.Flatten(),

    keras.layers.Dense(output_shape, use_bias=False),

])

# 24-quantized model
# model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=[input_shape]),
#     keras.layers.BatchNormalization(),
#     QActivation("quantized_bits(bits=24, integer=12)"),
#
#     qkeras.QDense(neuron_number1, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
#
#     keras.layers.BatchNormalization(),
#     QActivation("quantized_bits(bits=24, integer=12)"),
#
#     keras.layers.Reshape((9,1)),
#     QConv1D(1, 3,kernel_quantizer="quantized_bits(24,12)", use_bias=False),
#     # qkeras.QDense(neuron_number2, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
#     # keras.layers.Flatten(),
#
#     tf.keras.layers.Activation("sigmoid"),
#
#     keras.layers.BatchNormalization(),
#     QActivation("quantized_bits(bits=24, integer=12)"),
#
#     # keras.layers.Reshape((7, 1)),
#     QConv1D(1, 3, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
#     # qkeras.QDense(neuron_number3, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
#     keras.layers.Flatten(),
#
#     tf.keras.layers.Activation("relu"),
#
#     keras.layers.BatchNormalization(),
#     QActivation("quantized_bits(bits=24, integer=12)"),
#
#     qkeras.QDense(output_shape, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
#
#     keras.layers.BatchNormalization(),
#     QActivation("quantized_bits(bits=24, integer=12)"),
#
# ])

model.summary()

# qkeras.print_qstats(model)

model.compile(loss='mse', optimizer='Adam',metrics=['mae','mse'])

checkpoint_path = root_path+"Models/keras_post_soldering.h5"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose = 2, save_best_only=True)

epochs=250

history = model.fit(x=x_training_dataset, y=y_training_dataset, epochs=epochs,
                    callbacks=[checkpointer],
                    validation_data=(x_validation_dataset, y_validation_dataset),
                    )
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()

model.load_weights(checkpoint_path)

# Evaluate the model on test set
score = model.evaluate(x_test_dataset,y_test_dataset, verbose=0)

# Print test accuracy
print('\n', 'Test mae:', score)


fig, axs = plt.subplots(6)
for i in range(6):
    init=y_dataset[i,1]-y_dataset[i,0]
    init = (init - min) / (max - min)
    pred=model.predict(np.array([x_dataset[i],init.repeat(250,axis=0)]).transpose())
    axs[i].plot(pred.reshape(250))
    axs[i].plot((y_dataset[i] - min) / (max - min) )


x_dataset_sim = np.loadtxt(root_path+"Data/TestingData/DriftData_simulated_24bits.txt", delimiter=',')
y_dataset_sim = np.loadtxt(root_path+"Data/TestingData/ExpectedData_simulated_24bits.txt", delimiter=',')

j=1
accs=(y_dataset_sim[j,1]-x_dataset_sim[j,1])*100
accs=(accs-min)/(max - min)
accs=accs.repeat(250,axis=0)

pred_sim=model.predict(np.array([x_dataset[0],accs]).transpose())

pred_sim=pred_sim*(max-min)+min

valore_corretto=x_dataset_sim[j].reshape(243)-pred_sim.reshape(250)[0:243]/100

fig, ax = plt.subplots()
legend_properties = {'weight':'bold'}
plt.plot(x_dataset_sim[j].reshape(243), 'r',linewidth=2, label="Soldering Data")
plt.plot(y_dataset_sim[j].reshape(243),'b',linewidth=2, label="Expected Data")
plt.plot(valore_corretto, 'g',linewidth=2, label="Corrected Data")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
plt.legend(loc='best')
plt.legend(prop=legend_properties)
ax.legend(fontsize=15)
plt.xlabel("time [h]",fontsize=15) #,weight="bold"
plt.ylabel("Pressure [hPa]",fontsize=15)

print('\n', 'Mse(true values,drifted values):', mean_squared_error(y_dataset_sim[j].reshape(243), x_dataset_sim[j].reshape(243)))
print('\n', 'Mse(true values,corrected values):', mean_squared_error(y_dataset_sim[j].reshape(243), valore_corretto))

#
# soldering_mse=np.zeros(100)
# corrected_mse=np.zeros(100)
#
# for j in range(100):
#     accs = x_dataset_sim[j, 1] - y_dataset_sim[j, 1]
#     # accs = (accs - min) / (max - min)
#     accs = accs.repeat(250, axis=0)
#
#     pred_sim = model.predict(np.array([x_dataset[0], accs]).transpose())
#     # pred_sim = pred_sim * (max - min) + min
#
#     valore_corretto = x_dataset_sim[j].reshape(243) - pred_sim.reshape(250)[0:243] / 100
#     soldering_mse[j]=mean_squared_error(y_dataset_sim[j].reshape(243), x_dataset_sim[j].reshape(243))
#     corrected_mse[j]=mean_squared_error(y_dataset_sim[j].reshape(243), valore_corretto)
#
# print('\n', 'Mean(soldering_mse):',np.mean(soldering_mse))
# print( 'var(soldering_mse):',np.var(soldering_mse))
# print('max(soldering_mse):',np.max(soldering_mse))
# print( 'min(soldering_mse):',np.min(soldering_mse))
#
# print('\n\n', 'Mean(corrected_mse):',np.mean(corrected_mse))
# print('var(corrected_mse):',np.var(corrected_mse))
# print( 'max(corrected_mse):',np.max(corrected_mse))
# print('min(corrected_mse):',np.min(corrected_mse))
#
# worst_curve_index = np. where(corrected_mse == np.max(corrected_mse))
#
# j=worst_curve_index[0]
# accs=x_dataset_sim[j,1]-y_dataset_sim[j,1]
# # accs=(accs-min)/(max - min)
# accs=accs.repeat(250,axis=0)
# pred_sim=model.predict(np.array([x_dataset[0],accs]).transpose())
# # pred_sim=pred_sim*(max-min)+min
# valore_corretto=x_dataset_sim[j].reshape(243)-pred_sim.reshape(250)[0:243]/100
#
# fig, ax = plt.subplots()
# legend_properties = {'weight':'bold'}
# plt.plot(x_dataset_sim[j].reshape(243), 'r',linewidth=2, label="Soldering Data")
# plt.plot(y_dataset_sim[j].reshape(243),'b',linewidth=2, label="Expected Data")
# plt.plot(valore_corretto, 'g',linewidth=2, label="Corrected Data")
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.tick_params(axis='both', which='minor', labelsize=15)
# plt.legend(loc='best')
# plt.legend(prop=legend_properties)
# ax.legend(fontsize=15)
# plt.xlabel("time [h]",fontsize=15) #,weight="bold"
# plt.ylabel("Pressure [hPa]",fontsize=15)
#
# print('\n', 'Mse(true values,drifted values) corrected_worst:', mean_squared_error(y_dataset_sim[j].reshape(243), x_dataset_sim[j].reshape(243)))
# print('\n', 'Mse(true values,corrected values) corrected_worst:', mean_squared_error(y_dataset_sim[j].reshape(243), valore_corretto))
#
# print('\n', 'MEAN Acc(true values,drifted values) corrected_worst:', np.mean(np.abs(y_dataset_sim[j].reshape(243)- x_dataset_sim[j].reshape(243))))
# print( 'VAR Acc(true values,drifted values) corrected_worst:', np.var(np.abs(y_dataset_sim[j].reshape(243)- x_dataset_sim[j].reshape(243))))
# print( 'MAX Acc(true values,drifted values) corrected_worst:', np.max(np.abs(y_dataset_sim[j].reshape(243)- x_dataset_sim[j].reshape(243))))
#
# print('\n\n', 'MEAN Acc(true values,corrected values) corrected_worst:', np.mean(np.abs(y_dataset_sim[j].reshape(243)- valore_corretto)))
# print( 'VAR Acc(true values,corrected values) corrected_worst:', np.var(np.abs(y_dataset_sim[j].reshape(243)- valore_corretto)))
# print( 'MAX Acc(true values,corrected values) corrected_worst:', np.max(np.abs(y_dataset_sim[j].reshape(243)- valore_corretto)))