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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Doc """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, b in enumerate(boxes):
            bc = box_confidences[i]
            bcp = box_class_probs[i]

            bs = bc * bcp

            bcs = np.max(bs, axis=-1)
            bc1 = np.argmax(bs, axis=-1)

            idx = np.where(bcs >= self.class_t)

            filtered_boxes.append(b[idx])
            box_classes.append(bc1[idx])
            box_scores.append(bcs[idx])

        return np.concatenate(
            filtered_boxes), np.concatenate(
                box_classes), np.concatenate(box_scores)

    def iou(self, box1, box2):
        """Calculates the Intersection over Union (IoU) of two bounding boxes"""
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Performs non-maximum suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        classes = np.unique(box_classes)

        for c in classes:
            idx = np.where(box_classes == c)
            b = filtered_boxes[idx]
            bc1 = box_classes[idx]
            bs = box_scores[idx]

            ordered_indices = np.flip(bs.argsort(), axis=0)

            keep_indices = []

            while len(ordered_indices) > 0:
                maximum = ordered_indices[0]
                keep_indices.append(maximum)
                if len(ordered_indices) == 1:
                    break

                remaining = ordered_indices[1:]

                suppressed_indices = []
                for i in remaining:
                    xi1 = max(b[maximum, 0], b[i, 0])
                    yi1 = max(b[maximum, 1], b[i, 1])
                    xi2 = min(b[maximum, 2], b[i, 2])
                    yi2 = min(b[maximum, 3], b[i, 3])
                    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

                    box1_area = (b[maximum, 2] - b[maximum, 0]) * (b[maximum, 3] - b[maximum, 1])
                    box2_area = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
                    union_area = box1_area + box2_area - inter_area

                    iou = inter_area / union_area

                    if iou > self.nms_t:
                        suppressed_indices.append(i)

                ordered_indices = np.setdiff1d(remaining, suppressed_indices)

            box_predictions.append(b[keep_indices])
            predicted_box_classes.append(bc1[keep_indices])
            predicted_box_scores.append(bs[keep_indices])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores
