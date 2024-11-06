import unittest
import tensorflow as tf
import tensorflow_hub as hub
import video_extract
import numpy as np
import json

print("\n\n========== RUNNING TESTS. THIS WILL TAKE A WHILE.......... ==========\n\n")

with open("coco-labels-2014_2017.json", 'r') as file:
    coco_labels = json.load(file)

class TestDifferentModels(unittest.TestCase):

    def test_part_processing(self):
        ve = video_extract.VideoExtract()
        res = ve.extract_occurrences(ve.get_occurrences(8, 1))
        self.assertTrue(res > 0)  

    def test_slow_alternative_model(self):
        ve = video_extract.VideoExtract(detector = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1").signatures['default'])
        res = ve.extract_occurrences(ve.get_occurrences(22, 4))
        self.assertTrue(res > 0)

    def test_multi_part_processing(self):
        ve = video_extract.VideoExtract()
        res1 = video_extract.VideoExtract().get_occurrences(2,0)
        res2 = video_extract.VideoExtract().get_occurrences(2,1)
        self.assertTrue(ve.extract_occurrences(res1 + res2) > 0)

    def normalize_results(self, result):
        result["detection_scores"] = result["detection_scores"][0]
        detection_class_entities = []
        for dc in np.array(result["detection_classes"][0], dtype=int):
            detection_class_entities.append((list(filter(lambda label: label['id'] == dc, coco_labels))[0]["category"]).encode('ascii', 'strict'))
        result["detection_class_entities"] = detection_class_entities  
        return result

    def test_coco_model(self):
        ve = video_extract.VideoExtract(detected_object_classes = ["dog", "cat"], detector = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1"),dtype=tf.uint8, normalize_results=self.normalize_results)
        res = ve.extract_occurrences(ve.get_occurrences(22, 4))
        self.assertTrue(res > 0)

if __name__ == '__main__':
    unittest.main()