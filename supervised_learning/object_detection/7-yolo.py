#!/usr/bin/env python3
""" Doc """
import tensorflow.keras as K
import numpy as np
import os
from google.colab.patches import cv2_imshow
import cv2


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
        """Calculates the Intersection over
        Union (IoU) of two bounding boxes"""
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
        box_pdt = []
        pdt_box_classes = []
        pdt_box_sc = []

        classes = np.unique(box_classes)

        for c in classes:
            idx = np.where(box_classes == c)
            b = filtered_boxes[idx]
            bc1 = box_classes[idx]
            bs = box_scores[idx]

            ordered_indices = np.flip(bs.argsort(), axis=0)

            while len(ordered_indices) > 0:
                maximum = ordered_indices[0]
                box_pdt.append(b[maximum])
                pdt_box_classes.append(bc1[maximum])
                pdt_box_sc.append(bs[maximum])

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
                    box2_area = (
                        b[idx, 2] - b[idx, 0]) * (b[idx, 3] - b[idx, 1])
                    union_area = box1_area + box2_area - inter_area

                    ious[i] = inter_area / union_area

                ordered_indices = remaining[ious <= self.nms_t]

        box_pdt = np.array(box_pdt)
        pdt_box_classes = np.array(pdt_box_classes)
        pdt_box_sc = np.array(pdt_box_sc)

        return box_pdt, pdt_box_classes, pdt_box_sc

    @staticmethod
    def load_images(folder_path):
        """
        - folder_path: a string representing the
            path to the folder holding all the images to load
        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
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
                image, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            pimage = resized_image / 255
            pimages.append(pimage)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Displays the image with boxes, class names, and scores"""
        img_cp = image.copy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_index = box_classes[i]
            class_name = self.class_names[class_index]
            score = box_scores[i]

            # Draw box
            cv2.rectangle(img_cp, (
                int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Write class name and score
            text = f"{class_name} {score:.2f}"
            cv2.putText(
                img_cp, text, (
                    int(x1), int(
                        y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                            0, 0, 255), 1, cv2.LINE_AA)

        cv2_imshow(img_cp)
        key = cv2.waitKey(0)

        if key == ord('s'):
            output_dir = 'detections'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, img_cp)
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Predicts on all images in the folder and displays results"""
        images, image_paths = self.load_images(folder_path)
        predictions = []
        for image, path in zip(images, image_paths):
            image_name = os.path.basename(path)
            preprocessed_image, _ = self.preprocess_images([image])
            outputs = self.model.predict(preprocessed_image)
            boxes, box_confidences, box_class_probs = self.process_outputs(
                outputs, image.shape[:2])
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)
            box_pdt, pdt_box_classes, pdt_box_sc = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores)
            self.show_boxes(
                image, box_pdt, pdt_box_classes, pdt_box_sc, image_name)
            predictions.append((box_pdt, pdt_box_classes, pdt_box_sc))

        return predictions, image_paths
