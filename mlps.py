'''
    define MLPs models
'''

# first, import MultiLayerPerceptron from Machine Learning Extensions' Classifiers 
# then create three different MLPs to fit against datasets and compare its results

from mlxtend.classifier import MultiLayerPerceptron as MLP

# mlxtend.classifier.MultiLayerPerceptron Parameters Understanding

# eta => learning rate (between 0.0 and 1.0)
# epochs => passes over the training dataset
# print_progress => 
#   0: No output 
#   1: Epochs elapsed and cost 
#   2: Epochs elapsed, cost and time elapsed 
#   3: Epochs elapsed, cost and time elapsed and estimated time until completion
# random_seed => set random state for shuffling and initializing the weights.

mlp1 = MLP(hidden_layers=[10], 
          l2=0.00, 
          l1=0.0, 
          epochs=10, 
          eta=0.005, 
          momentum=0.0,
          decrease_const=0.0,
          minibatches=100, 
          random_seed=1,
          print_progress=3)

mlp2 = MLP(hidden_layers=[30], 
          l2=0.00, 
          l1=0.0, 
          epochs=10, 
          eta=0.05, 
          momentum=0.0,
          decrease_const=0.0,
          minibatches=100, 
          random_seed=1,
          print_progress=3)

mlp3 = MLP(hidden_layers=[60], 
          l2=0.00, 
          l1=0.0, 
          epochs=10, 
          eta=0.5, 
          momentum=0.0,
          decrease_const=0.0,
          minibatches=100, 
          random_seed=1,
          print_progress=3)