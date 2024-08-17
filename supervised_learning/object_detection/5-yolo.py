#!/usr/bin/env python3
""" Doc """
import numpy as np
import tensorflow.keras as K
import cv2
import os


class Yolo:
    """Yolo class for object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the Yolo class"""
        self.class_t = class_t
        self.nms_t = nms_t
        self.model = K.models.load_model(model_path)
        self.anchors = anchors
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.input_h = self.model.input.shape[1]
        self.input_w = self.model.input.shape[2]

    @staticmethod
    def sigmoid(x):
        """Applies the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Processes the outputs"""
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

            grid = np.tile(np.indices((
                g_w, g_h)).T, anchors.shape[0]).reshape(
                    (g_h, g_w) + anchors.shape)

            b_xy = (self.sigmoid(t_xy) + grid) / [g_w, g_h]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes based on class scores"""
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

            while len(ordered_indices) > 0:
                maximum = ordered_indices[0]
                box_predictions.append(b[maximum])
                predicted_box_classes.append(bc1[maximum])
                predicted_box_scores.append(bs[maximum])

                if len(ordered_indices) == 1:
                    break

                remaining = ordered_indices[1:]

                ious = np.zeros((len(remaining)))
                for i, idx in enumerate(remaining):
                    xi1 = max(b[maximum, 0], b[idx, 0])
                    yi1 = max(b[maximum, 1], b[idx, 1])
                    xi2 = min(b[maximum, 2], b[idx, 2])
                    yi2 = min(b[maximum, 3], b[idx, 3])
                    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

                    box1_area = (
                        b[maximum, 2] - b[maximum, 0]) * (
                            b[maximum, 3] - b[maximum, 1])
                    box2_area = (b[idx, 2] - b[idx, 0]) * (
                        b[idx, 3] - b[idx, 1])
                    union_area = box1_area + box2_area - inter_area

                    ious[i] = inter_area / union_area

                ordered_indices = remaining[ious <= self.nms_t]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder"""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                images.append(image)
                image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocesses a list of images for YOLO model"""
        input_h, input_w = self.input_h, self.input_w
        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])
            resized_image = cv2.resize(
                image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            pimage = resized_image.astype(np.float32) / 255.0  # Normalización

            pimages.append(pimage)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
