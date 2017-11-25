# Googlenet
Googlenet Implementation on DL frameworks

GoogLeNet is the network architecture that were created by the researchers at Google. As part of an ensemble of other trained similar models, it achieved a top-5 error rate of 6.67% on the [2014 ImageNet classification challenge](http://image-net.org/challenges/LSVRC/2014/results#clsloc).


input_shape: (224,224,3)

output: 1000 classes


---
## Architecture

![Googlenet Components](/doc/googlenet_components.png)

Googlenet consists of 4 components:
- stem
- inception modules
- auxiliary classifiers
- output classifier

![Googlenet incarnation](/doc/architecture.png)

Here are the parameters for each layer.




## TODOs
---

- [x] Implement it on Keras
- [ ] Test the model's performance
- [ ] Implement it on Tensorflow
- [ ] Visualize the test result
