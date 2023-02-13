from sgdp import SGDplus
from sgd_mn import SGDPlusMultiNeuron
from ADAM_on import myADAM
from ADAM_multi import myADAMMultiNeuron

import matplotlib.pyplot as plt

lr = 5*1e-3
# initialise SGD and SGD+ for one neuron
sgd_on = SGDplus(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = lr,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )


sgd_on.parse_expressions()
training_data = sgd_on.gen_training_data()
# SGD
sgd_on_loss_0 = sgd_on.train_one_neuron(training_data)
# SGD+ with mu = 0.5, 0.9
sgd_on_loss_5 = sgd_on.train_one_neuron(training_data, mu=0.5)
sgd_on_loss_9 = sgd_on.train_one_neuron(training_data, mu=0.9)

# initialise SGD and SGD+ for multi neuron
sgd_mn = SGDPlusMultiNeuron(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = lr,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

sgd_mn.parse_multi_layer_expressions()

# SGD 
sgd_mn_loss_0 = sgd_mn.train_multineuron(training_data)
# SGD+ with mu = 0.5, 0.9
sgd_mn_loss_5 = sgd_mn.train_multineuron(training_data, mu=0.5)
sgd_mn_loss_9 = sgd_mn.train_multineuron(training_data, mu=0.9)


# initialise ADAM for one neuron 
adam_on = myADAM(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = lr,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )


adam_on.parse_expressions()
# training_data = cgp.gen_training_data()

# loss_0 = cgp.train(training_data)
loss = adam_on.train(training_data)
# plt.plot(loss)

# loss_9 = cgp.train(training_data, mu=0.9)
# plt.plot(loss_0)
# plt.plot(loss_9)

# initialise ADAM for multi neuron 
adam_mn = myADAMMultiNeuron(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = lr,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

adam_mn.parse_multi_layer_expressions()

training_data = adam_mn.gen_training_data()
adam_mn_loss = adam_mn.train_multineuron(training_data)


# PLOT one neuron model losses
plt.plot(sgd_on_loss_0, label='SGD')
plt.plot(sgd_on_loss_5, label='SGD+, mu=0.5')
plt.plot(sgd_on_loss_9, label='SGD+, mu=0.9')
plt.plot(loss, label='ADAM')
plt.legend()
plt.title(f'SGD, SGD+ and ADAM for one neuron, LR={lr}')
plt.savefig(f"one_neuron_5")

# PLOT multi neuron model losses

plt.plot(sgd_mn_loss_0, label='SGD')
plt.plot(sgd_mn_loss_5, label='SGD+, mu=0.5')
plt.plot(sgd_mn_loss_9, label='SGD+, mu=0.9')
plt.plot(adam_mn_loss, label='ADAM')
plt.legend()
plt.title(f'SGD, SGD+ and ADAM for multi neuron, LR={lr}')
plt.savefig(f"multi_neuron_5")