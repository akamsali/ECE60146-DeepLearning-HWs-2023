from ComputationalGraphPrimer import ComputationalGraphPrimer
import operator
import random
import numpy as np
from tqdm import tqdm

class myADAMMultiNeuron(ComputationalGraphPrimer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def backprop_and_update_params_multineuron(self, y_error, class_labels):
            # backproped prediction error:
            pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
            pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
            for back_layer_index in reversed(range(1,self.num_layers)):
                input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
                input_vals_avg = [sum(x) for x in zip(*input_vals)]
                input_vals_avg = list(map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
                deriv_sigmoid =  self.gradient_vals_for_layers[back_layer_index]
                deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
                deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg, 
                                                                [float(len(class_labels))] * len(class_labels)))
                vars_in_layer  =  self.layer_vars[back_layer_index]                 ## a list like ['xo']
                vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## a list like ['xw', 'xz']

                layer_params = self.layer_params[back_layer_index]         
                ## note that layer_params are stored in a dict like        
                    ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
                ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
                transposed_layer_params = list(zip(*layer_params))         ## creating a transpose of the link matrix

                backproped_error = [None] * len(vars_in_next_layer_back)
                for k,varr in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] * 
                                                pred_err_backproped_at_layers[back_layer_index][i] 
                                                for i in range(len(vars_in_layer))])
    #                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
                pred_err_backproped_at_layers[back_layer_index - 1]  =  backproped_error
                input_vars_to_layer = self.layer_vars[back_layer_index-1]
                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j]
                    ##  Regarding the parameter update loop that follows, see the Slides 74 through 77 of my Week 3 
                    ##  lecture slides for how the parameters are updated using the partial derivatives stored away 
                    ##  during forward propagation of data. The theory underlying these calculations is presented 
                    ##  in Slides 68 through 71. 
                    for i,param in enumerate(layer_params):
                        
                        # @akamsali: update the velocity parameter and use 
                        g_t = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j] 

                        m_val = self.beta_1 * self.m[param] + (1-self.beta_1) * g_t
                        v_val = self.beta_2 * self.v[param] + (1-self.beta_2) * (g_t**2)
                        m_hat = m_val / (1 - self.beta_1 ** self.time[param] )
                        v_hat = v_val / (1 - self.beta_2 ** self.time[param] )

                        ## Update the learnable parameters
                        step = self.learning_rate * m_hat / np.sqrt(v_hat + self.epsilon)
                        self.vals_for_learnable_params[param] += step
                        # store the current values of first and second moment parameters 
                        # for next iteration of training
                        self.m[param] = m_val
                        self.v[param] = v_val
                        self.time[param] += 1 # update time step 

                ## Update the bias
                m_bias_val = self.beta_1 * self.m_bias[back_layer_index -1] + \
                            (1 - self.beta_1) * (np.sum(pred_err_backproped_at_layers[back_layer_index]) * np.mean(deriv_sigmoid_avg))
                v_bias_val = self.beta_2 * self.v_bias[back_layer_index -1] + \
                            (1 - self.beta_2) * (np.sum(pred_err_backproped_at_layers[back_layer_index]) * np.mean(deriv_sigmoid_avg)**2)

                m_bias_hat = m_bias_val / (1 - (self.beta_1 ** self.time_bias[back_layer_index -1]))
                v_bias_hat = v_bias_val / (1 - (self.beta_2 ** self.time_bias[back_layer_index -1]))
                
                ## Update the bias parameters
                bias_step = self.learning_rate * (m_bias_hat / np.sqrt(np.abs(v_bias_hat) + self.epsilon)) 
                # print(f"v_bias_hat: {v_bias_hat}")
                # np.sqrt(v_bias_hat + 1e-7)
                self.bias += bias_step

                # store the current values of first and second moment parameters 
                # for next iteration of training

                self.m_bias[back_layer_index -1] = m_bias_val
                self.v_bias[back_layer_index -1] = v_bias_val
                self.time_bias[back_layer_index -1] += 1 # update time step 
                        

    ######################################################################################################
    # @akamsali: modified func call name and take in momentum value \mu
    def train_multineuron(self, training_data, beta_1=0.9, beta_2=0.99, epsilon=1e-7):


        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data       
                batch = [batch_data, batch_labels]
                return batch                


        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}


        self.bias = [random.uniform(0,1) for _ in range(self.num_layers-1)]      ## Adding the bias to each layer improves 
                                                                                 ##   class discrimination. We initialize it 
                                                                                 ##   to a random number.
        # @akamsali: set hyperparameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon 
        # @akamsali: initialise learnable parameter moments
        self.m = {param: 0 for param in self.learnable_params}
        self.v = {param: 0 for param in self.learnable_params}
        self.time = {param: 1 for param in self.learnable_params}

        # @akamsali: initialise bias parameter moments
        self.time_bias = [1]*(self.num_layers-1)
        self.m_bias = [0]*(self.num_layers-1)
        self.v_bias = [0]*(self.num_layers-1)


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                          ##  Average the loss over iterations for printing out 
                                                                                 ##    every N iterations during the training loop.   
        for i in tqdm(range(self.training_iterations)):
            data = data_loader.getbatch()
            data_tuples = data[0]
            # print(data_tuples)
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                  ## FORW PROP works by side-effect 
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]      ## Predictions from FORW PROP
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]  ## Get numeric vals for predictions
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ## Calculate loss for batch
            loss_avg = loss / float(len(class_labels))                                         ## Average the loss over batch
            avg_loss_over_iterations += loss_avg                                              ## Add to Average loss over iterations
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))            ## Display avg loss
                avg_loss_over_iterations = 0.0                                                ## Re-initialize avg-over-iterations loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            # @akamsali: change to modified backprop
            self.backprop_and_update_params_multineuron(y_error_avg, class_labels)      ## BACKPROP loss
            
        return loss_running_record