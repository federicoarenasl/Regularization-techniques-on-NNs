# Training of our models built using the MLP ML Framework 
This blogpost will focus in going through the code used to train all our models for this study. We will first train the initial model and output all results to a ```.csv``` file, these results from the training will be later analysed and visualised in the ```Training-Results-Visualizer.mp``` part of our study. This is going to be a group of experiments divided in the following manner:
- 1. Experiments to identify the generalization problem in a Neural Network
- 2. Experiments to mitigate the problem using Dropout 
- 3. Experiments to mitigate the problem using L1 and L2 Regularisation techniques
- 4. Experiments to mitigate the problem using L1 Regularisation and Dropout
- 5. Training and testing the final model

The same structure will be kept on the ```Training-Results-Visualizer.md``` file, but instead of training progress data, we will show plots that showcase the performance of the model.

Before anything, let's import all necessary libraries, including the MLP ML framework, as well as the EMNIST data.
## 

## Initialization of input data and MLP framework
We import the mlp framework which will allow us to process the training and validation data we will use to fine-tune our models. We also import the different blocks that will allow us to build our Neural Network architectures, as well as the weight Initialization techniques, error function, learning rules and optimiser.


```python
import numpy as np
import logging
import pandas as pd
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider

# Seed a random number generator
seed = 11102019 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)

# Import mlp ML framework
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
```

    KeysView(<numpy.lib.npyio.NpzFile object at 0x2afdec97d0a0>)
    KeysView(<numpy.lib.npyio.NpzFile object at 0x2afdec97d0a0>)


We also define some helper functions that will allow us to train our model and organise the output from this training. We will use the ```export_values``` function to export the training progress of our networks to a ```.csv``` file.


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

def output_organiser(init_dict):
    final_dict = {}
    dict_keys = init_dict[1].keys()
    for i,layer in enumerate(dict_keys):
        final_dict[i] = []
        for key in init_dict.keys():
            final_dict[i].append(init_dict[key][layer])
    
    return final_dict

def export_values(stats_, keys_, runtime_, activations_, name):
    # Output training information
    train_valid_stats = {'error(train)': stats_[1:, keys_['error(train)']], 
                         'error(valid)': stats_[1:, keys_['error(valid)']],
                         'acc(train)': stats_[1:, keys_['acc(train)']],
                         'acc(valid)': stats_[1:, keys_['acc(valid)']], 
                         'runtime':[runtime_]*len(stats_[1:, keys_['error(train)']])}

    df_train_valid = pd.DataFrame(data=train_valid_stats)
    weight_info_df = pd.DataFrame(data=activations_)

    # Output weight information
    path = r"Results/"+name
    df_train_valid.to_csv(path+'_train_valid.csv')
    weight_info_df.to_csv(path+'_activations.csv')

def main_train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time, activations = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_title('Training and Validation Loss')
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_title('Training and Validation Accuracy')
    ax_2.set_xlabel('Epoch number')
    
    activations_dict = output_organiser(activations)
    
    return stats, keys, run_time, activations_dict
```

## 1. Experiments to identify the generalization problem in a Neural Network
We now start by training a very simple Network to identify the generalization problem in the Network. We will first train a model with a single hidden layer and 100 hidden units, but in the following part of the section we will also experiment with usinh 32, 62, and 128 hidden units, as well as 1, 2, and 3 hidden layers.


```python
# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_init, keys_init, run_time_init, activation_dict_init = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Export values for further analysis
export_values(stats_init, keys_init, run_time_init, activation_dict_init, 'init')
``` 
    Epoch 1: 3.0s to complete
        error(train)=8.21e-01, acc(train)=7.52e-01, error(valid)=8.39e-01, acc(valid)=7.44e-01

	.
	.
	.
    Epoch 100: 3.5s to complete
        error(train)=1.44e-01, acc(train)=9.40e-01, error(valid)=1.19e+00, acc(valid)=8.11e-01
   
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_6_302.png)
    
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_6_303.png)
    


## Understanding the influence of network width and depth on generalization performance of a vanilla Neural network
Let's continue our study by varying the shape of our network, in order to understand how this influences its performance.

### Varying the width of our network
Let's start by varying the number of hidden units. Our first model will have **32** hidden units, with the rest of the architecture intact.


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 32 # 32 Hidden Units fed into the ReLu activation function

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define Error function
error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_hu_32, keys_hu_32, run_time_hu_32, activation_dict_hu_32 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_hu_32, keys_hu_32, run_time_hu_32, activation_dict_hu_32, 'hu_32')
```

    Epoch 1: 1.9s to complete
        error(train)=1.16e+00, acc(train)=6.65e-01, error(valid)=1.17e+00, acc(valid)=6.63e-01

	.
	.
	.  

    Epoch 100: 2.1s to complete
        error(train)=4.81e-01, acc(train)=8.39e-01, error(valid)=6.63e-01, acc(valid)=7.98e-01


![png](Training-w-mlp-framework_files/Training-w-mlp-framework_8_302.png)
    
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_8_303.png)
    


We perform the same training for **64** hidden units.


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 64 # 64 Hidden Units fed into the ReLu activation function

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_hu_64, keys_hu_64, run_time_hu_64, activation_dict_hu_64 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_hu_64, keys_hu_64, run_time_hu_64, activation_dict_hu_64, 'hu_64')
```

    Epoch 1: 2.5s to complete
        error(train)=9.69e-01, acc(train)=7.14e-01, error(valid)=9.86e-01, acc(valid)=7.05e-01

	.
	.
	.

    Epoch 100: 2.5s to complete
        error(train)=2.53e-01, acc(train)=9.05e-01, error(valid)=7.73e-01, acc(valid)=8.12e-01


![png](Training-w-mlp-framework_files/Training-w-mlp-framework_10_302.png)
    

![png](Training-w-mlp-framework_files/Training-w-mlp-framework_10_303.png)
    


And then with **128** units. Remember, all of the results are being outputted to a ```.csv``` file, and will be analysed in the following blogpost.


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_hu_128, keys_hu_128, run_time_hu_128, activation_dict_hu_128 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_hu_128, keys_hu_128, run_time_hu_128, activation_dict_hu_128, 'hu_128')
```


    Epoch 1: 3.7s to complete
        error(train)=7.58e-01, acc(train)=7.72e-01, error(valid)=7.75e-01, acc(valid)=7.67e-01

	.
	.
	.
    Epoch 100: 3.9s to complete
        error(train)=9.37e-02, acc(train)=9.62e-01, error(valid)=1.59e+00, acc(valid)=8.15e-01

    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_12_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_12_303.png)
    


### Varying the depth of our network
We can now focus on varying the depth of our network, or the number of hidden layers that we have in the network. We will keep the number of hidden units to 100 throughout this phase of experiments.

Let's start by a network with **1** hidden layer.


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_hl_1, keys_hl_1, run_time_hl_1, activation_dict_hl_1 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_hl_1, keys_hl_1, run_time_hl_1, activation_dict_hl_1, 'hl_1')
```

    Epoch 1: 3.6s to complete
        error(train)=7.55e-01, acc(train)=7.70e-01, error(valid)=7.76e-01, acc(valid)=7.63e-01

	.
	.
	.
    Epoch 100: 4.0s to complete
        error(train)=9.61e-02, acc(train)=9.62e-01, error(valid)=1.54e+00, acc(valid)=8.17e-01

    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_14_302.png)
    
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_14_303.png)
    


Now let's icrease it to **2** hidden layers.


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_hl_2, keys_hl_2, run_time_hl_2, activation_dict_hl_2 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_hl_2, keys_hl_2, run_time_hl_2, activation_dict_hl_2, 'hl_2')
```



    Epoch 1: 4.9s to complete
        error(train)=7.04e-01, acc(train)=7.78e-01, error(valid)=7.29e-01, acc(valid)=7.72e-01
	.
	.
	.
    Epoch 100: 4.9s to complete
        error(train)=1.26e-01, acc(train)=9.54e-01, error(valid)=1.70e+00, acc(valid)=8.18e-01


![png](Training-w-mlp-framework_files/Training-w-mlp-framework_16_302.png)
    
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_16_303.png)
    


Finally, we train a model with **3** hidden layers.


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialize weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Define Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Define error function
error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_hl_3, keys_hl_3, run_time_hl_3, activation_dict_hl_3 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_hl_3, keys_hl_3, run_time_hl_3, activation_dict_hl_3, 'hl_3')
```


    Epoch 1: 5.0s to complete
        error(train)=6.77e-01, acc(train)=7.81e-01, error(valid)=7.04e-01, acc(valid)=7.72e-01
	.
	.
	.
    Epoch 100: 5.8s to complete
        error(train)=1.21e-01, acc(train)=9.54e-01, error(valid)=1.40e+00, acc(valid)=8.24e-01


    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_18_302.png)
    
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_18_303.png)
    


### 1.1. Baseline Model
We now decide to define a Baseline Model that counts with no Regularisation techinques. We will compare all models to this one, in the hopes of making the improvements evident as we employ more effective Regularisation techniques.

```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_bl, keys_bl, runtime_bl, activation_dict_bl = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_bl, keys_bl, runtime_bl, activation_dict_bl, 'bl')
```
```python

    Epoch 1: 4.7s to complete
        error(train)=6.75e-01, acc(train)=7.85e-01, error(valid)=7.04e-01, acc(valid)=7.78e-01

	.
	.
	.
    Epoch 100: 5.5s to complete
        error(train)=1.28e-01, acc(train)=9.52e-01, error(valid)=1.45e+00, acc(valid)=8.24e-01
```
    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_19_302.png)

    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_19_303.png)
    

## 2. Experiments to mitigate the problem using Dropout 
We now look at the effect of Dropout in our Baseline model. For this, we define three models with different Dropout inclusion probabilites at each time.

```python
from mlp.layers import DropoutLayer
from mlp.penalties import L1Penalty, L2Penalty
```



### Model 1
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

Dropout of **0.2** in all Hidden Layers


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Dropout random state
rng = np.random.RandomState(92019)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.2),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.2),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.2),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])
    
# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule() # Using the default learning rate of 0.001

# Remember to use notebook=False when you write a script to be run in a terminal
stats_d_02, keys_d_02, runtime_d_02, activation_dict_d_02 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_d_02, keys_d_02, runtime_d_02, activation_dict_d_02, 'd_02')
```


    Epoch 1: 6.6s to complete
        error(train)=3.50e+00, acc(train)=5.62e-02, error(valid)=3.50e+00, acc(valid)=5.65e-02
	.
	.
	.
    Epoch 100: 8.3s to complete
        error(train)=2.45e+00, acc(train)=2.74e-01, error(valid)=2.48e+00, acc(valid)=2.66e-01


![png](Training-w-mlp-framework_files/Training-w-mlp-framework_23_302.png)
    

    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_23_303.png)
    

### Model 2 
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

Dropout of **0.5** in all Hidden Layers


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Dropout random state
rng = np.random.RandomState(92019)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])
    
# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_d_05, keys_d_05, runtime_d_05, activation_dict_d_05 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_d_05, keys_d_05, runtime_d_05, activation_dict_d_05, 'd_05')
```
    Epoch 1: 6.1s to complete
        error(train)=2.14e+00, acc(train)=3.65e-01, error(valid)=2.14e+00, acc(valid)=3.64e-01
	.
	.
	.
    Epoch 100: 7.8s to complete
        error(train)=9.79e-01, acc(train)=6.96e-01, error(valid)=1.05e+00, acc(valid)=6.79e-01


![png](Training-w-mlp-framework_files/Training-w-mlp-framework_25_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_25_303.png)
    

### Model 3
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

Dropout  **in a decreasing fashion** 0.8, 0.5, 0.2


```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Dropout random state
rng = np.random.RandomState(92019)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.8),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.2),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])
    
# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_d_decf, keys_d_decf, runtime_d_decf, activation_d_decf = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_d_decf, keys_d_decf, runtime_d_decf, activation_d_decf, 'd_decf')

## 3. Experiments to mitigate the problem using L1 and L2 Regularisation techniques
We now use L2 Regularisation and L1 Regularisation to mitigate the problem.

## L2 Regularisation
```
    Epoch 1: 7.4s to complete
        error(train)=2.39e+00, acc(train)=2.85e-01, error(valid)=2.39e+00, acc(valid)=2.86e-01
	.
	.
	.
    Epoch 100: 8.1s to complete
        error(train)=9.21e-01, acc(train)=7.11e-01, error(valid)=1.04e+00, acc(valid)=6.88e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_27_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_27_303.png)
    

### Model 1
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

L2 regularisation penalty of  **0.0001** in all layers

  
```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-4)), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L2Penalty(1e-4))
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_L2_1e_4, keys_L2_1e_4, runtime_L2_1e_4, activation_L2_1e_4 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_L2_1e_4, keys_L2_1e_4, runtime_L2_1e_4, activation_L2_1e_4, 'L2_1e_4')
```

    Epoch 1: 5.4s to complete
        error(train)=6.99e-01, acc(train)=7.72e-01, error(valid)=7.19e-01, acc(valid)=7.67e-01

	.
	.
	.
    Epoch 100: 12.0s to complete
        error(train)=1.51e-01, acc(train)=9.39e-01, error(valid)=6.67e-01, acc(valid)=8.40e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_31_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_31_303.png)

### Model 2
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**
  
```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-2)), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L2Penalty(1e-2))
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_L2_1e_2, keys_L2_1e_2, runtime_L2_1e_2, activation_L2_1e_2 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_L2_1e_2, keys_L2_1e_2, runtime_L2_1e_2, activation_L2_1e_2, 'L2_1e_2')
```

L2 regularisation penalty of  **0.01** in all layers

    Epoch 1: 4.9s to complete
        error(train)=1.01e+00, acc(train)=7.01e-01, error(valid)=1.02e+00, acc(valid)=6.98e-01
	.
	.
	.
    Epoch 100: 5.8s to complete
        error(train)=7.75e-01, acc(train)=7.67e-01, error(valid)=7.90e-01, acc(valid)=7.63e-01


    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_33_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_33_303.png)

### Model 3
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

L2 regularisation penalty in  **increasing fashion** of 0.0001, 0.01, 0.1

  
```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-4)), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L2Penalty(0.1))
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_L2_incf, keys_L2_incf, runtime_L2_incf, activation_L2_incf = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_L2_incf, keys_L2_incf, runtime_L2_incf, activation_L2_incf, 'L2_incf')
```
    Epoch 1: 4.9s to complete
        error(train)=8.95e-01, acc(train)=7.32e-01, error(valid)=9.11e-01, acc(valid)=7.27e-01
	.
	.
	.
    Epoch 100: 10.3s to complete
        error(train)=4.41e-01, acc(train)=8.53e-01, error(valid)=5.20e-01, acc(valid)=8.28e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_35_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_35_303.png)

## L1 Regularisation


### Model 1
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

L2 regularisation penalty of  **0.0001** in all layers

```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L1Penalty(1e-4))
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_L1_1e_4, keys_L1_1e_4, runtime_L1_1e_4, activation_L1_1e_4 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_L1_1e_4, keys_L1_1e_4, runtime_L1_1e_4, activation_L1_1e_4, 'L1_1e_4')
```

    Epoch 1: 6.6s to complete
        error(train)=7.36e-01, acc(train)=7.65e-01, error(valid)=7.50e-01, acc(valid)=7.57e-01

	.
	.
	.
    Epoch 100: 6.5s to complete
        error(train)=3.26e-01, acc(train)=8.80e-01, error(valid)=4.33e-01, acc(valid)=8.52e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_38_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_38_303.png)
    

### Model 2
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

L2 regularisation penalty of  **0.01** in all layers
    

```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L1Penalty(1e-2))
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_L1_1e_2, keys_L1_1e_2, runtime_L1_1e_2, activation_L1_1e_2 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_L1_1e_2, keys_L1_1e_2, runtime_L1_1e_2, activation_L1_1e_2, 'L1_1e_2')
```

    Epoch 1: 6.4s to complete
        error(train)=3.85e+00, acc(train)=2.17e-02, error(valid)=3.85e+00, acc(valid)=1.96e-02
	.
	.
	.
    Epoch 100: 7.0s to complete
        error(train)=3.85e+00, acc(train)=2.17e-02, error(valid)=3.85e+00, acc(valid)=1.98e-02


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_40_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_40_303.png)

### Model 3
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

L2 regularisation penalty in  **increasing fashion** of 0.0001, 0.01, 0.1

   
```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialisation of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Definition of the Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)), 
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L1Penalty(0.1))
])

# Definition of the error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_L1_incf, keys_L1_incf, runtime_L1_incf, activation_L1_incf = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_L1_incf, keys_L1_incf, runtime_L1_incf, activation_L1_incf, 'L1_incf')
```

    Epoch 1: 6.5s to complete
        error(train)=3.85e+00, acc(train)=2.15e-02, error(valid)=3.85e+00, acc(valid)=2.01e-02

	.
	.
	.
    Epoch 100: 6.3s to complete
        error(train)=3.85e+00, acc(train)=2.17e-02, error(valid)=3.85e+00, acc(valid)=2.01e-02


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_42_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_42_303.png)

## 4. Experiments to mitigate the problem using L1 Regularisation and Dropout
For this part of the study, we use L1 Regularisation, since it shows better performance than L2 Regularisation, and we combine it with Dropout to investigate its effect on the generalization performance.


### Model 1
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

Dropout of **0.2** and L2 regularisation penalty of **0.001**

```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialization of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Dropout random state
rng = np.random.RandomState(92019)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)), 
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.8),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.8),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.8),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-2)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L1Penalty(1e-2))
])

# Definition of error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_d02_L2_1e3, keys_d02_L2_1e3, runtime_d02_L2_1e3, activation_d02_L2_1e3 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_d02_L2_1e3, keys_d02_L2_1e3, runtime_d02_L2_1e3, activation_d02_L2_1e3, 'd08_L1_1e2')
```
    Epoch 1: 9.3s to complete
        error(train)=3.85e+00, acc(train)=2.14e-02, error(valid)=3.85e+00, acc(valid)=2.19e-02
	.
	.
	.
    Epoch 100: 8.4s to complete
        error(train)=3.85e+00, acc(train)=2.17e-02, error(valid)=3.85e+00, acc(valid)=2.01e-02


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_45_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_45_303.png)
    

### Model 2
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

Dropout of **0.5** and L2 regularisation penalty of **0.01**

```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialization of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Dropout random state
rng = np.random.RandomState(92019)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)), 
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.5),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L1Penalty(1e-4))
])

# Definition of error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_d05_L2_1e2, keys_d05_L2_1e2, runtime_d05_L2_1e2, activation_d05_L2_1e2 = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_d05_L2_1e2, keys_d05_L2_1e2, runtime_d05_L2_1e2, activation_d05_L2_1e2, 'd05_L1_1e4')
```


    Epoch 1: 7.7s to complete
        error(train)=2.50e+00, acc(train)=2.62e-01, error(valid)=2.49e+00, acc(valid)=2.63e-01

	.
	.
	.
    Epoch 100: 7.9s to complete
        error(train)=1.21e+00, acc(train)=6.27e-01, error(valid)=1.23e+00, acc(valid)=6.21e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_47_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_47_303.png)
    

### Model 3
Number of ReLu hidden units: **128**

Number of Hidden layers:**3**

Dropout **in a decreasing fashion** and L2 regularisation **in an increasing fashion**
   

```python
# Reset Data Providor
train_data.reset()
valid_data.reset()

# Setup hyperparameters
learning_rate = 0.05
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 128 # 128 Hidden Units fed into the ReLu activation function

# Initialization of weights and biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Dropout random state
rng = np.random.RandomState(92019)

# Definition of Neural Network Model
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)), 
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.8),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.7),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    DropoutLayer(rng, incl_prob=0.7),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L1Penalty(1e-4)),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L1Penalty(0.1))
])

# Definition of error function
error = CrossEntropySoftmaxError()

# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
stats_ddecf_L2incf, keys_ddecf_L2incf, runtime_ddecf_L2incf, activation_ddecf_L2incf = main_train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Output results for further anlysis
export_values(stats_ddecf_L2incf, keys_ddecf_L2incf, runtime_ddecf_L2incf, activation_ddecf_L2incf, 'ddecf_L1incf')
```
    Epoch 1: 8.6s to complete
        error(train)=2.42e+00, acc(train)=2.98e-01, error(valid)=2.43e+00, acc(valid)=2.97e-01
	.
	.
	.
    Epoch 100: 7.7s to complete
        error(train)=9.90e-01, acc(train)=6.92e-01, error(valid)=1.04e+00, acc(valid)=6.78e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_49_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_49_303.png)

## 5. Training and testing the final model
We choose **Model 2** form the Dropout + L1 Regularisation to report our best results. We load the test data, and provide this one during training instead of the validation data


```python
test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)
```

    Epoch 1: 6.7s to complete
        error(train)=7.96e-01, acc(train)=7.50e-01, error(valid)=8.44e-01, acc(valid)=7.35e-01
	.
	.
	.
    Epoch 100: 6.5s to complete
        error(train)=3.41e-01, acc(train)=8.76e-01, error(valid)=4.66e-01, acc(valid)=8.40e-01


    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_52_302.png)
    



    
![png](Training-w-mlp-framework_files/Training-w-mlp-framework_52_303.png)
    

