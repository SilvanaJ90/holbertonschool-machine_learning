#!/usr/bin/env python3
""" Doc """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Doc """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ The initializes the Yolo class """
        self.class_t = class_t
        self.nms_t = nms_t
        self.model = K.models.load_model(model_path)
        self.anchors = anchors
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f.readlines()]
