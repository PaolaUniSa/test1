import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
import qkeras
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings

from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from qkeras.utils import model_quantize

# ##################################################
# Ensure reproducibility of random operations:
np.random.seed(123)

rng = tf.random.Generator.from_seed(123)
tf.random.set_global_generator(rng)
#####################################################################################
# Dataset
# x_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/Drift Analysis LPS22HH/DATASET/Matlab_dataset_high_temperature/x_dataset_high_temperature.txt", delimiter=',')
# y_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset_high_temperature/Dataset_rawData/error_dataset_high_temperature.txt", delimiter=',')

# y_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset_high_temperature/Dataset_rawData/error_dataset_high_temperature_Class1_DUT4_DUT9.txt", delimiter=',')

y_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset_high_temperature/Dataset_rawData/error_dataset_high_temperature_Class2_DUT1_DUT12_DUT18.txt", delimiter=',')

# y_dataset = np.loadtxt("C:/Unisa/AI in pressure sensor/DATASET/Matlab_dataset_high_temperature/Dataset_rawData/error_dataset_high_temperature_Class3_DUT23.txt", delimiter=',')

# min=-2048
# max=2047.99
# y_dataset=(y_dataset-min)/(max-min)
y=y_dataset

input_shape=1 # (tempo)
output_shape=1

x=np.arange(y_dataset.shape[1])
time=np.tile(x,(y_dataset.shape[0],1))

time=time.flatten()
# min=np.min(time)
# max=np.max(time)
# time=(time-min)/(max-min)


y=y.flatten()
# min=np.min(y_dataset)
# max=np.max(y_dataset)
# y_dataset=(y_dataset-min)/(max-min)
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
    # qkeras.QDense(neuron_number3, kernel_quantizer="quantized_bits(24,12)", use_bias=False),
    tf.keras.layers.Activation("relu"),
    keras.layers.Flatten(),

    keras.layers.Dense(output_shape, use_bias=False),

])

model.summary()


model.compile(loss='mse', optimizer='ADAM',metrics=['mae','mse'])




checkpoint_path = "C:/Unisa/AI in pressure sensor/Models/keras_2h_at_100C_Class2.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose = 2, save_best_only=True)#,monitor='loss', mode='min')


# ##################################################
# batch_size=256
epochs=4

history = model.fit(x=time[0:int(0.8*time.shape[0])], y=y[0:int(0.8*time.shape[0])], epochs=epochs,
                    # batch_size=30,
                    callbacks=[checkpointer],
                    validation_data=(time[int(0.8*time.shape[0]):int(0.9*time.shape[0])], y[int(0.8*time.shape[0]):int(0.9*time.shape[0])]),
                    # shuffle=True
                    )

# plt.plot(history.history["loss"], label="Training Loss")#validation_y
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()

model.load_weights(checkpoint_path);
# Evaluate the model on test set
score = model.evaluate(time[int(0.9*time.shape[0]):time.shape[0]],y[int(0.9*time.shape[0]):time.shape[0]], verbose=0)

# Print test accuracy
print('\n', 'Test mae:', score)



plt.figure(2)
for i in range(y_dataset.shape[0]):
    plt.plot(y_dataset[random.randint(0, y_dataset.shape[0]-1), :])

error_predicted=model(time[0: y_dataset.shape[1]])
plt.plot(error_predicted,c='b',linewidth=5.0)

np.savetxt("C:/Unisa/AI in pressure sensor/Models/errors/error_predicted_Class1.txt",error_predicted.numpy())

#print detailed statistic of model
qkeras.print_qstats(model)
