# Reinforcement Learning Environment
A Reinforcement Learning environment which is based on a sheep eating grass logic.   

### Quick Start Guide
From the cloned repository, run the following commands in the terminal:

$ conda env create -f environment.yml  
$ conda activate sheep_env

If using pycharm, set the interpreter to the python version in the created conda env e.g:

.../anaconda3/envs/sheep_env/bin/python

There are two ways to test the environment. Run the "suite_manual.py" file to test manually or run "suite_random.py" to test random behaviour.   

When adding or removing a dependency from the environment.yml list, run:  
$ conda env update --file environment.yml

To run Tensorboard enter:  
$ tensorboard --logdir=/path/to/output/logs/folder/

### Used Sources/Dependencies

#### Machine Learning with Phil
https://www.youtube.com/watch?v=5fHngyN8Qhw

#### sentdex
https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
https://www.youtube.com/watch?v=t3fbETsIBCY  
https://www.youtube.com/watch?v=qfovbG84EBg

### System Dependencies 
TBD