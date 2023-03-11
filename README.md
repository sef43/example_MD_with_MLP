# Simple example of creating, training, and using a Machine Learning Potential for Molecular Dynamics


This repo contains an example of creating, training, and using a machine learning potential. For simplicity a 2D Lennard Jones system will be used as the dataset. 

The main notebook is [simple_mlp.ipynb](simple_mlp.ipynb)

The explains the steps and runs all of them.


The scripts it uses are in three separate python files:

- [generate.py](generate.py) - creates the dataset
- [train_mlp.py](train_mlp.py) - defines the MLP and trains it
- [md_with_mlp.py](md_with_mlp.py) - uses the MLP to run MD


A further notebook: [MD_with_autograd/md_with_autograd.ipynb](/MD_with_autograd/md_with_autograd.ipynb)
gives an example of using autograd to compute the force from a molecular potential energy function.
