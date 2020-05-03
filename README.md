# unsupervised-object-detection-with-rpn
## Setup Details 
This repository is developed and tested with python 3.7.3. If you are missing python header files, install python 3.7-dev

Run the setup script   
`./setup.sh`

## Visualize
Start the visdom server before starting the training  
`python -m visdom.server`


## Train
Start another new browser window from GCP instance.

Run the train command

`python train.py --gpu_ids -1 --batch_size 15 --display_winsize 10 --game SkiingNoFrameskip-v0 --dynamic_datagen
