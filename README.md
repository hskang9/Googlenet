# Googlenet
Googlenet Implementation on DL frameworks

GoogLeNet is the network architecture that were created by the researchers at Google. As part of an ensemble of other trained similar models, it achieved a top-5 error rate of 6.67% on the [2014 ImageNet classification challenge](http://image-net.org/challenges/LSVRC/2014/results#clsloc).


## input_shape: (224,224,3)

## output: 1000 classes


---
## Architecture

![Googlenet Components](/doc/googlenet_components.png)

Googlenet consists of 4 components:
### stem
stem layer is the sequential chain of convolution, pooling, and local response normalization operations, similar to AlexNet. Stem layer is referred in later papers. The authors of later papers cite technical issues with it when compared with pure network of inception modules, so it may be disappear.

### inception modules(which its inspiration comes from the meme 'we need to go deeper')
inception module is the basic building block of GoogLeNet. It is a set of convolutions and poolings at different scales,   each done in parallel, then concatenated together with depth. Along the way, 1x1 convolutions(3x3 reduce, 5x5 reduce) are used to reduce the dimensionality of inputs to convolutions with larger filter sizes(3x3, 5x5). This approach results in a high performing model with drastically fewer parameters. As a result, GoogLeNet has a 12 times lower training parameters than AlexNet.
  
### auxiliary classifiers for ensemble learning
Given the relatively large depth of the network, the ability to back-propagate gradient through all layer is a concern. On the other hand, the researchers thought that the features produced by layers in the middle of the network should be very discriminative. By adding auxiliary classifiers connected to these intermediate layers(inception modules), encouraging discrimination between classes is expected, increasing backpropagation signal and eventually solving "vanishing gradient" problem. During training, their loss gets added to the total loss of the network with a discount weight(the losses of the auxiliary classifiers were weighted by 0.3)

### output classifier for final classification
The output classifier is the component which classifies the image. It performs an average pooling operation followed by a softmax activation on a fully connected layer.



![Googlenet incarnation](/doc/architecture.png)

Here are the parameters for each layer.




## TODOs
---

- [x] Implement it on Keras
- [ ] Build train/test pipeline
- [ ] Test the model's performance
- [ ] Implement it on Tensorflow
- [ ] Visualize the test result
