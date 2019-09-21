# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
import tflearn as TFLEARN


class BaseModel(object):
	def build_graph(self, input, expected, reuse):
		raise NotImplementedError()
