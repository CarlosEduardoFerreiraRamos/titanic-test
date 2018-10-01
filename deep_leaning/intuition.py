"""
#Atificial neural net

imput layer     hidden layers     outputlayer
                neuron
neuron
                neuron              neuron
neuron
                neuron              neuron
neuron
                neuron
"""

"""
The neuron
the  Activation function
How do Neural networks work
How do neural networks learn
Gradient Descent
Stochastic Gradient Descent
Backpropagation
"""

"""
#The Neuron:

(independent varibles)
                    synapse
input value      ----------|
                           |
input value      ----------|-----Neuron -------->   Output value
                           |                        they can be: 
input value      ----------|                        * continuos(like price)
                                                    * binary (will exit yes/no)
                                                    * Categorical ( **)

** they more than one output, they should be like dummyvariables 

Input:
Values (x)
one input for the only one roll??

Synapses: 
to they are added weights(w)

Neuron:

*Sum
Ws = (x*w)i ... of all

*activation function
0(Ws) 
"""


"""
#Activation function
there are types

Threshold function
    0 (x) = { 1 if x >= 0
            { 0 if x < 0

Sigmoid function
    0 (x) = 1 / (1+ e^-x)

Rectifier function
    0 (x) = max(x,0)
    *one of the most used functions in artificial networks

Hyperbolic Tangent fucntion (tanh)
    0 (x) = (1 -e^-2x) / (1 + e^-2x)

Paper: Deep Sparse rectifier neural networks => tells why the Rectifier function is one of most used functions.

each layer can have a different function

"""

"""
#How do neural networks?
cost function:
C = 0.5 * (y_pred - y)^2

*semns to be a diff fro the prediction and real value
*One perception for row??

Cost fucntion of all rows:
C all = Sum of 0.5 * (y_pred - y)^2

Article about cross function:
CrossVaidated 2015
"""

"""
#Gradient Descent
For what I gatter, Gradient Descent is the analises of curve slope direction.
Determianting if its decresing or incresing.

* the cost fucntion needs to be convex
* adjust the weights after interate through all the rolls
* deterministic alghorith
"""

"""
#Stochastic Gradient Descent

* adjust the weights after each rolls
* as you puck the rows, possible at randon, different values may appear
"""
# Artice Gradient Descent:

# A neural Network in 13 lines of Python (part 2 -gradient Descent)
# Andrew Trask (2015) 

# Neural Networks and Deep Learning
# Michael Neilsen (2015)

"""
#Backpropagation
Neural Networks and Deep Learning
Michael Neilsen (2015)

"""

"""
Step by step trainig Atificial neural net with stochastic gradient descent:

1 -  Randomly initialise the weights to small number close to 0, but not Zero

2 - Input the first observation of your dataset in the input layer each feature in one imput node

3 - Forward Propagation -> from left to right, the neuron are activated in a way taht the imapctof each
    neuron's activation is limited by the weights. Propagate the activation until getting the predicted
    reult(y_pred)

4 - Compare the predicted result to the actual result. Measure generated error.

5 - Back propagation: from right to left, the erro is back propagated. Update the weights according to
    how much they are responsible for the error. The learning rate decides by how much we update the weights

6 - Reapeat steps 1- 5 and update the weights after each observation (Reinforce Learning)
    OR
    Reapeat steps 1- 5 but update the weights only after a batch of observation (Batch leaning)

7 - when the whole training set passed through the ANN, taht makes as epach. Redo more epochs
"""