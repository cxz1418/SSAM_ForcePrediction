# SSAM_ForcePrediction
This is a Tensorflow implementation for Inferring Force Estimation without Haptic Sensor. This repository includes the implementation of CNN-LSTM module as well.

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
We also constructed a model for learning and generating a db composed of sequential images for interaction force estimation
https://github.com/hyeon-jo/Interaction-force-estimation-based-on-deep-learning

## How to use
VGG-like-model  based examples are included. 

## Pretrained-model 
