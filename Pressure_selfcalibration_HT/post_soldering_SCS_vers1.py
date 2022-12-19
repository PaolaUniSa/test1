# @authors: Paola Vitolo
# @Date: 04/07/2022
# @Brief: calibrare il post soldering con un modello simile a quello per calibrare l'effetto di esposizione prolungata ad alte temperature


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
# Ensure reproducibility of random operations:
np.random.seed(123)

rng = tf.random.Generator.from_seed(123)
tf.random.set_global_generator(rng)

#####################################################################################
# Dataset for training
x_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/x_acc_in_ore_80devices.txt", delimiter=',') # shape=80x250 time (0->249)
y_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/y_acc_in_ore_80devices.txt", delimiter=',') # shape=80x250 accuracy of the 80 devices in post-soldering

# Normalization between 0 and 1
# min=np.min(y_dataset)
# max=np.max(y_dataset)


# Normalization between 0 and 1
# 24 bits -> 2^24=16,777,216 values -> [-2^23 ; 2^23-1]/sensitivity= [-8388608 , 8388607] /[2^12/hPa] --> min=-2^23/2^12 and max=(2^23-1)/2^12=2047.99
min=-2048
max=2047.99
y_dataset=(y_dataset-min)/(max-min)

# x_dataset=(x_dataset-min)/(max-min)

# y_dataset=(y_dataset*(pow(2,24)-1)) - pow(2,23)
# y_dataset=y_dataset/(pow(2,12))


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
    keras.layers.BatchNormalization(),
    QActivation("quantized_bits(bits=24, integer=12)"),

    qkeras.QDense(neuron_number1, kernel_quantizer="quantized_bits(24,12)", use_bias=False),

    keras.layers.BatchNormalization(),
    QActivation("quantized_bits(bits=24, integer=12)"),

    keras.layers.Reshape((9,1)),
    QConv1D(1, 3,kernel_quantizer="quantized_bits(24,12)", use_bias=False),
    # qkeras.QDense(neuron_number2, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
    # keras.layers.Flatten(),

    tf.keras.layers.Activation("sigmoid"),

    keras.layers.BatchNormalization(),
    QActivation("quantized_bits(bits=24, integer=12)"),

    # keras.layers.Reshape((7, 1)),
    QConv1D(1, 3, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
    # qkeras.QDense(neuron_number3, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
    keras.layers.Flatten(),

    tf.keras.layers.Activation("relu"),

    keras.layers.BatchNormalization(),
    QActivation("quantized_bits(bits=24, integer=12)"),

    qkeras.QDense(output_shape, kernel_quantizer="quantized_bits(24,12)", use_bias=False),

    keras.layers.BatchNormalization(),
    QActivation("quantized_bits(bits=24, integer=12)"),

])

model.summary()

qkeras.print_qstats(model)


model.compile(loss='mse', optimizer='Adam',metrics=['mae','mse'])
    # (loss='MeanAbsoluteError',
    #          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),#'adam',
    #          metrics=['MeanAbsoluteError'])




checkpoint_path = "C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/post_soldering_quantizedModel_final_ver2.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose = 2, save_best_only=True)#,monitor='loss', mode='min')
#
#
# # ##################################################
# # batch_size=256
epochs=50
# x=x.transpose()
# y=y.transpose()


#
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
# x_dataset_sim = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/x_dataset_30sim_1h.txt", delimiter=',')
# y_dataset_sim = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/y_dataset_30sim_1h.txt", delimiter=',')


x_dataset_sim = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/x_dataset_100sim_1h_quantized.txt", delimiter=',')
y_dataset_sim = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset/y_dataset_100sim_1h_quantized.txt", delimiter=',')

j=1
accs=x_dataset_sim[j,1]-y_dataset_sim[j,1]
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


soldering_mse=np.zeros(100)
corrected_mse=np.zeros(100)

for j in range(100):
    accs = x_dataset_sim[j, 1] - y_dataset_sim[j, 1]
    accs = (accs - min) / (max - min)
    accs = accs.repeat(250, axis=0)

    pred_sim = model.predict(np.array([x_dataset[0], accs]).transpose())
    pred_sim = pred_sim * (max - min) + min

    valore_corretto = x_dataset_sim[j].reshape(243) - pred_sim.reshape(250)[0:243] / 100
    soldering_mse[j]=mean_squared_error(y_dataset_sim[j].reshape(243), x_dataset_sim[j].reshape(243))
    corrected_mse[j]=mean_squared_error(y_dataset_sim[j].reshape(243), valore_corretto)

print('\n', 'Mean(soldering_mse):',np.mean(soldering_mse))
print( 'var(soldering_mse):',np.var(soldering_mse))
print('max(soldering_mse):',np.max(soldering_mse))
print( 'min(soldering_mse):',np.min(soldering_mse))

print('\n\n', 'Mean(corrected_mse):',np.mean(corrected_mse))
print('var(corrected_mse):',np.var(corrected_mse))
print( 'max(corrected_mse):',np.max(corrected_mse))
print('min(corrected_mse):',np.min(corrected_mse))

worst_curve_index = np. where(corrected_mse == np.max(corrected_mse))

j=worst_curve_index[0]
accs=x_dataset_sim[j,1]-y_dataset_sim[j,1]
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

print('\n', 'Mse(true values,drifted values) corrected_worst:', mean_squared_error(y_dataset_sim[j].reshape(243), x_dataset_sim[j].reshape(243)))
print('\n', 'Mse(true values,corrected values) corrected_worst:', mean_squared_error(y_dataset_sim[j].reshape(243), valore_corretto))

print('\n', 'MEAN Acc(true values,drifted values) corrected_worst:', np.mean(np.abs(y_dataset_sim[j].reshape(243)- x_dataset_sim[j].reshape(243))))
print( 'VAR Acc(true values,drifted values) corrected_worst:', np.var(np.abs(y_dataset_sim[j].reshape(243)- x_dataset_sim[j].reshape(243))))
print( 'MAX Acc(true values,drifted values) corrected_worst:', np.max(np.abs(y_dataset_sim[j].reshape(243)- x_dataset_sim[j].reshape(243))))

print('\n\n', 'MEAN Acc(true values,corrected values) corrected_worst:', np.mean(np.abs(y_dataset_sim[j].reshape(243)- valore_corretto)))
print( 'VAR Acc(true values,corrected values) corrected_worst:', np.var(np.abs(y_dataset_sim[j].reshape(243)- valore_corretto)))
print( 'MAX Acc(true values,corrected values) corrected_worst:', np.max(np.abs(y_dataset_sim[j].reshape(243)- valore_corretto)))