# HydraNet

The HydraNet is a neural network architecture that splits into multiple branches (or heads) close to the top of the network. These heads are trained individually and learn different things by having each mini batch of training data be weighted differently for the different heads. We average over the outputs from the different heads and this gives us a final result.

We use existing state of the art architectures and make modifications to them accordingly. During training, the part of the network lying before these heads (the model body) have their weights frozen, and we train only the heads. After a certain number of epochs we un-freeze the body of the network and continue training.

Details and visualisations of the training runs and models can be found [here](https://app.wandb.ai/raghav1810/hydranet-temp/reports?view=raghav1810%2FReport%20%230).

![Test accuracies graph](https://raw.githubusercontent.com/raghav1810/HydraNet/master/W%26Btest_acc_chart.png)

### Architectures used
- Resnet110
- Preresnet110
- DensenetBC (k=12 L=100)

### Datasets
- CIFAR10
- CIFAR100

### Implementation details
- Python 3.6
- Pytorch 1.1.0

Experiment tracking was done using [Wandb](https://www.wandb.com).

Amazon EC2 spot instances used for compute power. (p2.8xlarge, p2.xlarge)

### Usage
Example run:
```
python run.py -arch preresnet -d cifar100 --split_pt 50 --n_heads 8 --lr 0.01 --epochs 64  --unfreeze 56 --schedule 48 --train-batch 128 --test-batch 128

```

### Example
A torchviz visualization of a resnet with 4 heads (n_heads=4) splitting at point 53 (split_pt=53) in the network would look like [this](https://github.com/raghav1810/HydraNet/blob/master/model_eg_graph.svg)

### Acknowledgements
This project was done guidance of Dr Roland Baddeley ([link](http://www.bris.ac.uk/expsych/people/roland-j-baddeley/index.html)) at the School of Psychological Science, University of Bristol.
