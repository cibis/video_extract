
# video_extract
Extract and compile in a new video all occurrences of certain object type using python tensorflow object detection models

The module is using TensorFlow hub models to search through the video for occurrences of certain object classes, it extracts them and compile in a new video

Default hub model used is(600 boxable categories)
https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1

Section Links : [Constructor](#constructor) , [Tests](#tests), [Coming-Next](#coming-next)
# Constructor

```python
ve = video_extract.VideoExtract(detected_object_classes = ["dog", "cat"])
```
Default constructor parameters
```python
                 detector = hub.load("hub url").signatures['default'], 
                 dtype=tf.float32, 
                 detected_object_classes = ["Dog", "Cat"], 
                 minimum_detection_score=0.6, 
                 source_video="./source/source.mp4", 
                 result_video="./output/result.mp4", 
                 inspection_rate_in_seconds=0.20, 
                 maximum_missed_detections_time_in_seconds=3, 
                 minimum_video_part_size_in_seconds=1,  
                 video_tmp_output_path="./_runtime/", 
                 normalize_results = None
```
- inspection_rate_in_seconds: how often are the frames extracted for inspection
- maximum_missed_detections_time_in_seconds: for how long to try to detect the classes when compiling occurrences
- minimum_video_part_size_in_seconds; what is the minimum occurrences length

To use
```python
ve = video_extract.VideoExtract(detected_object_classes = ["dog", "cat"])
ve.extract_all_occurrences()
```
or if only a part of the video needs to be inspected

```python
ve = video_extract.VideoExtract(detected_object_classes = ["dog", "cat"])
#split in two and get second part
ve.extract_occurrences(ve.get_occurrences(2, 1))
```
this approach can be also used for distributed processing

Check test.py for more usage examples

# Tests
```
python -m unittest -v test
```

# Coming-Next
- Adding an example of distributed processing using Kubernetes