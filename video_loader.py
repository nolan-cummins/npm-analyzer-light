from PySide6.QtCore import *
import os
import cv2
from time import sleep
from videotools import placeLabel, frameDifferencing, nearestOdd
import numpy as np

class processVideos(QThread):
    frame_out = Signal(np.ndarray)
    contours_out = Signal(object, str)
    name_out = Signal(str, str)
    
    def __init__(self):
        super().__init__()

        self.cap = None
        self.fps = 1
        self.running = True
        self.pause = False
        self.display_fps = 30
        self.name = ''
        self.file = ''
        self.init_mutex = QMutex()
        self.pause_mutex = QMutex()
        self.settings_mutex = QMutex()
        self.settings={}

    def loadVideo(self, file):
        try:
            with QMutexLocker(self.init_mutex):
                self.cap = cv2.VideoCapture(file)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.name = os.path.basename(file)
                self.file = file
            print(f"Loading video: {self.name}")
        except Exception as e:
            print(f"Error loading video {self.name}: {e}")

    def clearVideo(self):
        with QMutexLocker(self.init_mutex):
            self.cap = None

    def capExists(self):
        with QMutexLocker(self.init_mutex):
            return self.cap is not None
    
    def videoState(self, paused):
        with QMutexLocker(self.pause_mutex):
            self.pause = paused

    @Slot(dict)
    def loadSettings(self, settings):
        with QMutexLocker(self.settings_mutex):
            self.settings = settings
    
    @Slot()
    def restartVideo(self):
        with QMutexLocker(self.init_mutex):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def applyFilters(self, frame):
        boxes, centers, boxes2D = [], [], []
        with QMutexLocker(self.settings_mutex):
            settings = self.settings

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        if settings["blurToggle"]:
            frame = cv2.GaussianBlur(frame, (settings["blurVal"], settings["blurVal"]), 0)
        
        if settings["adaptToggle"]:
            method = settings["adaptMethod"]
            if method == "Mean":
                frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                              settings["adaptArea"], settings["adaptValueC"])
            elif method == "Gaussian":
                frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              settings["adaptArea"], settings["adaptValueC"])
                
        if settings["thresholdToggle"]:
            if settings["autoToggle"]:
                otsu, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_normal = int(otsu / 255 * 100)
            else:
                _, frame = cv2.threshold(frame, settings["thresholdVal"], 255, cv2.THRESH_BINARY)
        
        if settings["subBackToggle"]:
            method = settings["subBackMethod"]
            if method == "MOG2":  # Mixture of Gaussians 2
                frame = settings["subBackModels"][method].apply(frame, learningRate=0.001)
            if method == "KNN":  # K Nearest Neighbors
                frame = settings["subBackModels"][method].apply(frame, learningRate=0.01)
            _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if settings["embossToggle"]:
            kernel = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
            frame = cv2.convertScaleAbs(cv2.filter2D(frame, -1, kernel) * settings["embossVal"])
        
        if settings["invertToggle"]:
            frame = cv2.bitwise_not(frame)
        
        if settings["dilationToggle"]:
            frame = cv2.dilate(frame, None, iterations=settings["dilateVal"])
            kernel = np.ones((3, 3), np.uint8)
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

        if settings["frameDiffToggle"]:
            boxes, centers, boxes2D = frameDifferencing(frame, settings["frameDiffValue"], settings["frameDiffValueMax"])

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame, boxes, centers, boxes2D, settings
    
    def run(self):
        tick_freq = cv2.getTickFrequency()
        fps_tick = 0
        last_tick = 0
        frame_num = 0
        run_once = True
        detected_objects={}
        while self.running:
            if self.cap is not None:
                with QMutexLocker(self.init_mutex):
                    cap = self.cap
                    display_time = 1/self.display_fps
                    frame_time = 1/self.fps
                try:
                    with QMutexLocker(self.pause_mutex):
                        pause = self.pause
                    if not pause:
                        start_tick = cv2.getTickCount()
                        ret, frame = cap.read()
                        if fps_tick == 0:
                            fps_tick = cv2.getTickCount()
                        if last_tick == 0:
                            last_tick = cv2.getTickCount()
                        if not ret:
                            if not settings["batchRecord"]:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            else:
                                with QMutexLocker(self.init_mutex):
                                    self.contours_out.emit(detected_objects, self.name)
                                    self.name_out.emit(self.name, self.file)
                                return
                        
                        original = frame.copy()
                        frame, boxes, centers, boxes2D, settings = self.applyFilters(frame)

                        current_tick = cv2.getTickCount()
                        if (current_tick - fps_tick) / tick_freq >= display_time:
                            fps_tick = current_tick
                            if frame is not None:
                                if settings["showOriginal"]:
                                    frame = original
                                if len(boxes) > 0 and settings["drawContours"]:
                                    cv2.drawContours(frame,boxes,-1,(0, 0, 255),2)
                                if settings["showFPS"]:
                                    if current_tick - last_tick > 0:
                                        fps = tick_freq / (current_tick - last_tick);
                                        msgFPS = f'FPS : {str(int(fps))}'
                                        frame = placeLabel(frame, msgFPS, 1.5, 1, (5, 5))

                                self.frame_out.emit(frame)

                        last_tick = cv2.getTickCount()
                        if not settings["batchRecord"]:
                            if not run_once:
                                run_once = True
                            process_time = (cv2.getTickCount() - start_tick) / tick_freq
                            sleep_time = max(frame_time - process_time, 0)
                            sleep(sleep_time)
                        else:
                            if run_once:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_num = 0
                                run_once = False
                            detected_objects[frame_num]=boxes2D
                            frame_num+=1
                except Exception as e:
                    print(e)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()