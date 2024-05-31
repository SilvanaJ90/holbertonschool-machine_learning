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

    @staticmethod
    def sigmoid(x):
        """
            Doc
        """
        return (1. / (1. + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """ Doc """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            g_h, g_w = output.shape[:2]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            sigmoid_conf = self.sigmoid(output[..., 4])
            sigmoid_prob = self.sigmoid(output[..., 5:])

            box_conf = np.expand_dims(sigmoid_conf, axis=-1)
            box_class_prob = sigmoid_prob

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.input.shape.as_list()[1:3]

            grid = np.tile(np.indices(
                (g_w, g_h)).T,
                anchors.shape[0]).reshape((g_h, g_w) + anchors.shape)

            b_xy = (self.sigmoid(t_xy) + grid) / [g_w, g_h]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
