from ComputationalGraphPrimer import ComputationalGraphPrimer
import random
import numpy as np
import operator


class SGDplus(ComputationalGraphPrimer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.momentum = mu

    def backprop_and_update_params(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        @akamsali:
        modified from the backprop_and_update_params_one_neuron_model method in the ComputationalGraphPrimer class
        in the ComputationalGraphPrimer.py file.

        The modification is to use the SGD+ algorithm to update the step size

        from:
        p_{t+1} = p_t - \eta g_{t+1}

        to:
        v_{t+1} = \mu v_t + g_{t+1}
        p_{t+1} = p_t - \eta v_{t+1}

        where \mu is the momentum parameter
        \eta = learning rate
        g_{t+1} = gradient of the loss function with respect to the learnable parameters (deriv_sigmoid)
        p_t = learnable parameters
        v_0 = all 0's
        """

        input_vars = self.independent_vars
        vals_for_input_vars_dict = dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i, param in enumerate(self.vals_for_learnable_params):
            ## Calculate the next step in the parameter hyperplane

            self.v[i] = self.momentum * self.v[i] + (
                y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid
            )
            step = self.learning_rate * self.v[i]
            ## Update the learnable parameters
            self.vals_for_learnable_params[param] += step

        ## Update the bias
        self.v_bias = self.learning_rate * y_error * deriv_sigmoid + (
            self.momentum * self.v_bias
        )
        self.bias += self.v_bias

    def train(self, training_data, mu=None):
        """
        @akamsali: Taking Avi's code as is for training a one neuron model.  The only modification
        is to return the loss running record so that we can plot it later.
        """

        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        if mu is None:
            self.momentum = 0
        else:
            self.momentum = mu
        self.vals_for_learnable_params = {
            param: random.uniform(0, 1) for param in self.learnable_params
        }

        self.bias = random.uniform(
            0, 1
        )  ## Adding the bias improves class discrimination.
        ##   We initialize it to a random number.

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)

            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [
                    (item, 0) for item in self.training_data[0]
                ]  ## Associate label 0 with each sample
                self.class_1_samples = [
                    (item, 1) for item in self.training_data[1]
                ]  ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):
                cointoss = random.choice(
                    [0, 1]
                )  ## When a batch is created by getbatch(), we want the
                ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)

            def getbatch(self):
                batch_data, batch_labels = (
                    [],
                    [],
                )  ## First list for samples, the second for labels
                maxval = 0.0  ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval:
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [
                    item / maxval for item in batch_data
                ]  ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = (
            0.0  ##  Average the loss over iterations for printing out
        )
        self.v = [0] * (
            len(self.vals_for_learnable_params)
        )  ## Initialize the velocity vector to all 0's
        self.v_bias = 0
        ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids = self.forward_prop_one_neuron_model(
                data_tuples
            )  ##  FORWARD PROP of data
            loss = sum(
                [
                    (abs(class_labels[i] - y_preds[i])) ** 2
                    for i in range(len(class_labels))
                ]
            )  ##  Find loss
            loss_avg = loss / float(len(class_labels))  ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg
            if i % (self.display_loss_how_often) == 0:
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0  ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(
                map(
                    operator.truediv,
                    data_tuple_avg,
                    [float(len(class_labels))] * len(class_labels),
                )
            )
            self.backprop_and_update_params(
                y_error_avg, data_tuple_avg, deriv_sigmoid_avg
            )  ## BACKPROP loss
        # plt.figure()
        return loss_running_record
        # plt.show()
