# Flint Implementation
Flint: A Novel Methodology of Training Sample Quality Assessment and its Application to Fixing Deep Neural Network Models

## Usage

### Train the DL model

An example of training the DL model is as follows:

`python train.py -m <model name> -d <dataset> -v <version> -e <epochs>`

> `-m`   : `mobilenetv2`, `resnet18`, `resnet34`,  `shufflenetv2`
> `-d`   : `cifar10`, `svhn`
> `-v`   : the version number.
> `-e`   : number of epochs for training.

### Fix the DL model by Flint

An example of fixing the DL model by Flint is as follows:

`python main.py -m <model name> -d <dataset> -v <version> -e <epochs> -s <strategy> -r1 <rate 1> -r2 <rate 2>`

> `-m`   : `mobilenetv2`, `resnet18`, `resnet34`,  `shufflenetv2`
> `-d`   : `cifar10`, `svhn`
> `-v`   : an integer
> `-e`   : loading model trained after the given epochs.
> `-v`   : the version number.
> `-s`   : strategy of Flint (1, 2, and 3).
> `-r1`   : floating number of r1.
> `-r2`   : floating number of r2.
