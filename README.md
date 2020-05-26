# MTL
Multi Task Learning with MGDA - UP described in [Sener et al's 2019 paper](https://arxiv.org/pdf/1810.04650.pdf).

**Disclaimer:** So far, only the MTL with linear proxy task loss and Single Task training options are implemented.



## Setup

The conda environment can be recreated using the following command.

`conda create -f environment.yaml`

To use the conda environment in a Jupyter notebook, run:

`python -m ipykernel install --user --name mtl_env --display-name "Python (mtl_env)"`


## MTL Approach to CAPTCHA Prediction

The multi-task learning based approach is tested on the prediction of CAPTCHA characters. In each CAPTCHA image, the recognition of 5 different CAPTCHA characters can be rephrased as a multi-label and multi-task classification problem.


To run the multi-task learning training run the command:

`python train.py -c config.json`

To run the single-task learning training, update the config with `"task_type":"single"` and run the above Python call.




**Note:** The number of CAPTCHA images are scaled up by custom data augmentation of the images. 
