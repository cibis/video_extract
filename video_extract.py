import tensorflow as tf
import tensorflow_hub as hub
import cv2
import shutil
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from os.path import exists
import glob
from pathlib import Path
import math
import time
from yaspin import yaspin

class VideoExtract:

    def __init__(self, 
                 detector = hub.load("https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1").signatures['default'], 
                 dtype=tf.float32, 
                 detected_object_classes = ["Dog", "Cat"], 
                 minimum_detection_score=0.6, 
                 source_video="./source/source.mp4", 
                 result_video="./output/result.mp4", 
                 inspection_rate_in_seconds=0.20, 
                 maximum_missed_detections_time_in_seconds=3, 
                 minimum_video_part_size_in_seconds=1,  
                 video_tmp_output_path="./_runtime/", 
                 normalize_results = None):
        self.detected_object_classes = detected_object_classes
        self.minimum_detection_score = minimum_detection_score
        self.source_video = source_video
        self.result_video = result_video        
        self.inspection_rate_in_seconds = inspection_rate_in_seconds
        self.maximum_missed_detections_time_in_seconds = maximum_missed_detections_time_in_seconds
        self.minimum_video_part_size_in_seconds = minimum_video_part_size_in_seconds
        self.video_tmp_output_path = video_tmp_output_path
        self.frame_path = video_tmp_output_path + "frame.jpg"
        self.duration,self.video_frames_per_second = self.get_video_duration()
        self.detector = detector
        self.dtype = dtype
        self.normalize_results = normalize_results
        if exists(result_video):
            os.remove(result_video)
        self.spinner = yaspin(color="green")

        if os.path.exists(video_tmp_output_path) and os.path.isdir(video_tmp_output_path):
            shutil.rmtree(video_tmp_output_path)    

        os.makedirs(video_tmp_output_path)

        if os.path.exists(os.path.dirname(result_video)):
            shutil.rmtree(os.path.dirname(result_video))   

        os.makedirs(os.path.dirname(result_video))

    def get_video_duration(self):
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(self.source_video)
        duration       = round(clip.duration)
        fps            = round(clip.fps)
        return duration, fps

    def get_video_part_length(self, part_num):
        duration, fps = self.get_video_duration()
        return math.floor(duration / part_num)
    
    def __hasClasses(self, class_names, scores, checkClasses, min_score=0.1):
        for ix, class_name in enumerate(class_names):
            if scores[ix] >= min_score and class_name.decode("ascii").lower() in [x.lower() for x in checkClasses]:                    
                return True, class_name.decode("ascii"), int(100 * scores[ix])
        return False, "", 0

    def __load_img(self, path): 
        # imgFile = cv2.copyMakeBorder(cv2.imread(path) , 0, 640-360, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
        # cv2.imwrite(path, imgFile)         
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img        
    
    def __run_detector(self):
        img = self.__load_img(self.frame_path)

        converted_img  = tf.image.convert_image_dtype(img, self.dtype)[tf.newaxis, ...]

        result = self.detector(converted_img)

        result = {key:value.numpy() for key,value in result.items()}
        
        if self.normalize_results:
            result=self.normalize_results(result)

        return self.__hasClasses(result["detection_class_entities"], result["detection_scores"], self.detected_object_classes, self.minimum_detection_score)

    def __appendVideoPortion(self, firstFrameWithContent, lastFrameWithContent):
        clip = VideoFileClip(self.source_video)
        clip = clip.subclip(firstFrameWithContent / self.video_frames_per_second, lastFrameWithContent / self.video_frames_per_second)
        clip.write_videofile(self.video_tmp_output_path + str(firstFrameWithContent) + ".mp4")
        clip.close()

    def get_occurrences(self, part_num, part_index):
        try:
            print(f"get_occurrences part_num: {part_num}, part_index: {part_index}")
            self.spinner.start()
            vidcap = cv2.VideoCapture(self.source_video)
            success,image = vidcap.read()
            frameIndex = 0
            occurrences = []
            part_length = math.floor(self.get_video_part_length(part_num) * self.video_frames_per_second)
            print(f"frames per second: {self.video_frames_per_second}")
            print(f"total source length(seconds): {self.duration}")
            print(f"part_length(seconds): {self.get_video_part_length(part_num)}")
            print(f"part_length(frames): {part_length}, total number of frames: {self.duration*self.video_frames_per_second},   range to process {part_index * part_length} to {(part_index + 1) * part_length}")            
            while success:  
                cv2.imwrite(self.frame_path, image)
                if frameIndex == part_index * part_length:
                    print(f"processing part_num: {part_num}")
                if frameIndex >= part_index * part_length and frameIndex < (part_index + 1) * part_length and frameIndex%(self.video_frames_per_second * self.inspection_rate_in_seconds) == 0:
                    m, s = divmod(int(frameIndex / self.video_frames_per_second), 60)
                    h, m = divmod(m, 60)
                    #print(f'{h:d}:{m:02d}:{s:02d}  frameIndex: {frameIndex}')  
                    detected, classDetected, score = self.__run_detector()
                    if detected:     
                        print(f'\n{h:d}:{m:02d}:{s:02d},  frameIndex: {frameIndex}, class: {classDetected}, score: {score}')
                        occurrences.append(frameIndex)
                if frameIndex == (part_index + 1) * part_length - 1:
                    print(f"finished processing part_num: {part_num}")
                    break
                frameIndex += 1
                success,image = vidcap.read()
            self.spinner.stop()
            print(f"occurrences {occurrences}")
        except:
            raise
        finally:
            self.spinner.stop()
        return occurrences
    
    def extract_occurrences(self, occurrences):
        contentBlockStart = -1
        contentEndBlock = -1
        if not occurrences:
            print("No occurrences detected") 
            return 0
        for frameIndex in occurrences:
            if(contentBlockStart == -1):
                contentBlockStart = frameIndex
                contentEndBlock = frameIndex
            else:
                if(contentEndBlock > -1 and frameIndex - contentEndBlock > 0 and frameIndex - contentEndBlock >= self.video_frames_per_second * self.maximum_missed_detections_time_in_seconds):
                    print(f'did not pass minimum missed detections check diff: {(frameIndex - contentEndBlock)/self.video_frames_per_second}s  frameIndex:{frameIndex:d}')
                    if(contentBlockStart > -1 and contentEndBlock - (self.video_frames_per_second * self.minimum_video_part_size_in_seconds) >= contentBlockStart):
                        print(f'append video {(contentEndBlock - contentBlockStart)/self.video_frames_per_second}s  contentBlockStart: {contentBlockStart:d}; contentEndBlock:{contentEndBlock:d}; frameIndex:{frameIndex:d}') 
                        self.__appendVideoPortion(contentBlockStart, contentEndBlock)
                    else:
                        print(f'did not pass the minimum video segment length check diff: {(contentEndBlock - contentBlockStart)/self.video_frames_per_second}s frameIndex:{frameIndex:d}')
                    contentBlockStart = frameIndex
                    contentEndBlock = frameIndex
                else:
                    contentEndBlock = frameIndex
            diff=""
            if(contentBlockStart != -1 and contentEndBlock != -1): 
                diff=f'{(contentEndBlock - contentBlockStart)/self.video_frames_per_second}s'
            print(f'contentBlockStart: {contentBlockStart:d}; contentEndBlock:{contentEndBlock:d}; frameIndex:{frameIndex:d}, {diff}') 
        if(contentBlockStart > -1 and contentEndBlock - (self.video_frames_per_second * self.minimum_video_part_size_in_seconds) >= contentBlockStart):
            print(f'append video {(contentEndBlock - contentBlockStart)/self.video_frames_per_second}s  contentBlockStart: {contentBlockStart:d}; contentEndBlock:{contentEndBlock:d}; frameIndex:{frameIndex:d}') 
            self.__appendVideoPortion(contentBlockStart, contentEndBlock)
        else:
            print(f'did not pass the minimum video segment length check diff: {(contentEndBlock - contentBlockStart)/self.video_frames_per_second}s frameIndex:{frameIndex:d}')
        file_names = []

        video_file_list = glob.glob(f"{self.video_tmp_output_path}*.mp4")

        for video in video_file_list:    
            file_names.append(int(Path(video).stem)) 

        file_names.sort()

        loaded_video_list = []

        video_cnt = len(file_names)

        for idx, video in enumerate(file_names):
            print(f"Adding video file:{self.video_tmp_output_path + str(video)}.mp4  {idx} out of {video_cnt}")
            try:
                loaded_video_list.append(VideoFileClip(self.video_tmp_output_path + str(video) + ".mp4"))
            except Exception as error:
                print("An exception occurred:", error) 

        if not loaded_video_list:
            print("No occurrences detected") 
        else:
            final_clip = concatenate_videoclips(loaded_video_list)

            final_clip.write_videofile(self.result_video) 
            final_clip.close()

            for video in loaded_video_list:
                video.close()

        if os.path.exists(self.video_tmp_output_path) and os.path.isdir(self.video_tmp_output_path):
            shutil.rmtree(self.video_tmp_output_path) 
        return video_cnt        
        
    def extract_all_occurrences(self):
        self.extract_occurrences(self.get_occurrences(1, 0))