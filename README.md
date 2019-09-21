# SSAM_ForcePrediction
This is a Tensorflow implementation for Inferring Force Estimation without Haptic Sensor. This repository includes the implementation of CNN-LSTM Baseline module. (https://arxiv.org/pdf/1811.07190.pdf)

## Abstract
Humans can approximately infer the force of interaction
between objects using only visual information because we
have learned it through experiences. Based on this idea,
in this paper, we propose a method based on a recurrent
convolutional neural network that uses sequential images
to infer the interaction force without using a haptic sensor To train and validate deep learning methods, we collected a
large number of images and corresponding data concerning the interaction forces between objects shown therein
through an electronic motor-based device. To focus on the
changing appearances of a target object owing to external
force in the images, we develop a sequential image-based
attention module that learns a salient model from temporal dynamics for predicting unknown interaction forces. We
propose a sequential image-based spatial attention module and a sequential image-based channel attention module, which are extended to exploit multiple images based on
corresponding weighted average pooling layers. Extensive
experimental results verified that the proposed method can
successfully infer interaction forces in various conditions
featuring different target materials, changes in illumination,
and directions of external forces.

![Alt text](/samples/Fig_main.JPG)

## Prerequisites
* Python 3.x
* TensorFlow 1.x
* tflearn


## Prepare Data set
### Download DB
We also constructed a model for learning and generating a db composed of sequential images for interaction force estimation
https://github.com/hyeon-jo/Interaction-force-estimation-based-on-deep-learning
### TF-Record
In order to optimize the step of loading files in training and validation, we should convert the dataset (sequential video images) into TF-Records.
https://www.tensorflow.org/tutorials/load_data/tf_records

## Train a Model
You can find example of training script in train_queue.py.
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_queue.py</code></pre>
Some critical arguments (Config.py):
* *BATCH_SIZE*: train batch size.
* *TRAIN_LEARNING_RATE*: train learning rate.
* *MODEL_OPTIMIZER*: train optimizer parameter.
* *MODEL_NETWORK*: The CNN_architecture constucted in nets/cnn.py.


## Evaluate a Model
Below script gives you an example of evaluating a model with CNN_LSTM after training.
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py</code></pre>

## How to use
VGG-like-model  based examples are included. 



## Pretrained-model 
