import ast
import numpy as np
from collections import defaultdict
from time import time
import os
import pandas as pd
import cv2

def convertFromStrings(contours: dict):
    print("Converting strings to Python types...")
    contours_fixed = {}
    for frame_index in contours:
        boxes2D = contour_data[frame_index]
        new_boxes2D = []
        for box2D in boxes2D:
            try:
                new_boxes2D.append(ast.literal_eval(box2D))
            except Exception as e:
                if "nan" in str(e):
                    new_boxes2D.append(np.nan)
                else:
                    print("Error parsing .csv")
        contours_fixed[int(frame_index)] = new_boxes2D
    df_contours = pd.DataFrame(contours_fixed)
    return df_contours

class detectedObject():
    def __init__(self, identity, new_box, closest_box, frame_index, velocities, area_new, area_closest, centers):
        self.identity = identity
        self.box2D = {}
        self.frames_in = []
        self.velocities = {}
        self.areas = {}
        self.positions = {}
        self.addDetection(closest_box, frame_index-1, velocities, area_new, centers[0])
        self.addDetection(new_box, frame_index, velocities, area_closest, centers[1])
        
    def addDetection(self, box2D, frame, velocities, area, center):
        self.box2D[frame]=box2D
        self.velocities[frame] = velocities
        self.areas[frame] = area
        self.positions[frame] = center
        self.frames_in.append(frame)

def trackObjects(contours, min_len=30):
    object_id = 0
    tracked_objects, filtered_objects = {}, []
    prev_objects = []
    temp_objects = []
    
    for frame_index in range(1, len(contours.keys())):
        new_boxes2D = contours[frame_index]
        old_boxes2D = contours[frame_index-1]
        old_centers = []
        old_boxes2D_clean = []
        for old_box2D in old_boxes2D:
            if not isinstance(old_box2D, float) and np.array(old_box2D[0]).shape == (2,):
                old_centers.append(old_box2D[0])
                old_boxes2D_clean.append(old_box2D)

        old_boxes2D = old_boxes2D_clean
        for new_box2D in new_boxes2D:
            if not isinstance(new_box2D, float):
                if np.shape(old_centers) == (0,):
                    break
                exists=False
                distances = np.sum((np.array(old_centers) - new_box2D[0]) ** 2, axis=1)
                closest_box = old_boxes2D[np.argmin(distances)]
    
                threshold = np.max(np.concatenate((new_box2D[1], closest_box[1])))**2
                distance = np.linalg.norm(np.asarray(new_box2D[0]) - np.asarray(closest_box[0]))
                if distance < threshold:
                    result, _ = cv2.rotatedRectangleIntersection(new_box2D, closest_box) # check if overlapping
                else:
                    result = 0
    
                if result > 0:
                    centers = np.array([new_box2D[0],closest_box[0]])   
                    velocities = np.diff(centers, axis=0)[0]
                    area = np.prod(new_box2D[1])
                    if len(tracked_objects) == 0:
                        print("Initializing first object-match pair")
                        new_object = detectedObject(object_id, new_box2D, closest_box, frame_index, velocities, area, np.prod(closest_box[1]), centers)
                        tracked_objects[new_object]=None
                        prev_objects.append(new_object)
                        object_id += 1
                        continue
                    for tracked_object in prev_objects:
                        if (frame_index - 1) in tracked_object.box2D and tracked_object.box2D[frame_index-1] == closest_box:
                            #print(f"Found match with {tracked_object.identity}", end='\r')
                            tracked_object.addDetection(new_box2D, frame_index, velocities, area, centers[0])
                            prev_objects.remove(tracked_object)
                            temp_objects.append(tracked_object)
                            tracked_objects[tracked_object] = None
                            old_boxes2D.remove(closest_box)
                            old_centers.remove(closest_box[0])
                            exists=True
                            break
                    if not exists:
                        #print(f"Adding new object {object_id}", end='\r')
                        new_object = detectedObject(object_id, new_box2D, closest_box, frame_index, velocities, area, np.prod(closest_box[1]), centers)
                        temp_objects.append(new_object)
                        object_id += 1
        print(f"Total objects found: {object_id} ({frame_index/(len(contours.keys())-1)*100:.2f}%)", end='\r')
        prev_objects = temp_objects.copy()
        temp_objects.clear()

    print(f"\nFiltering by minimum length: {min_len}")
    for tracked_object in tracked_objects.keys():
        if len(tracked_object.box2D) >= 30:
            filtered_objects.append(tracked_object)
    print(f"Final length: {len(filtered_objects)}")
    return filtered_objects

def extractData(detect_objects):
    data=defaultdict(list)
    for object_detected in detect_objects:
        for frame in object_detected.frames_in:
            data[frame].append(str([object_detected.velocities[frame].tolist(),
                               object_detected.areas[frame].tolist(),
                               object_detected.positions[frame].tolist()]))
    max_len = max(len(objects) for objects in data.values()) 
    
    for frame, values in data.items():
        while len(values) < max_len:
            values.append(np.nan)
                
    return pd.DataFrame(data)

def byFrame(detect_objects):
    contours_by_frame=defaultdict(list)
    ids_by_frame=defaultdict(list)
    for object_detected in detect_objects:
        for frame in object_detected.frames_in:
            box2D = object_detected.box2D[frame]
            box = cv2.boxPoints(box2D)
            box = np.intp(box)
            contours_by_frame[frame].append(box)
            ids_by_frame[frame].append(object_detected.identity)

    return contours_by_frame, ids_by_frame

def topRight(pts):
    pts = np.array(pts)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_right = pts[np.argmin(diff)] 
    return top_right