import sys
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
import pyqtgraph as pg
import numpy as np
import pandas as pd
from time import *
import os
import cv2
import traceback
import ctypes
import warnings
import traceback
from collections import defaultdict, deque
import inspect

myappid = 'nil.npm.pyqt.3' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

from importlib import reload
import json

import ui_light
reload(ui_light)
from ui_light import *

from scaleBar import scaleBar
from videotools import ScaleBar, nearestOdd
from qtools import *
from video_loader import processVideos

class MainWindow(QMainWindow, Ui_MainWindow):
    settings_signal = Signal(tuple)
    save_signal = Signal(str)
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initialized = False
        self.setupUi(self)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle("NPM Video Analyzer (light)")
        
        self.videoDisplay = QLabel(self)
        self.videoDisplay.setGeometry(QRect(34, 85, 640, 480))
        self.blank = QPixmap("background.jpg").scaled(self.videoDisplay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.videoDisplay.setPixmap(self.blank)
        self.videoDisplay.setAlignment(Qt.AlignCenter)
        self.pauseButton.clicked.connect(self.pauseDisplay)
        self.playButton.clicked.connect(self.resumeDisplay)
        self.recordButton.clicked.connect(self.emitSettings)
        self.videoLoader = None
        
        self.actionEdit.triggered.connect(self.onEditScaleBar)
        self.actionLoadDirectory.triggered.connect(self.loadDirectorySelect)
        self.actionSaveDirectory.triggered.connect(self.saveDirectorySelect)
        self.actionOpen_Video.triggered.connect(self.openVideo)
        self.actionClear.triggered.connect(self.clearVideos)
        self.actionFPS.triggered.connect(self.emitSettings)
        self.actionOriginal_Frame.triggered.connect(self.emitSettings)
        self.actionBounding_Boxes_2.triggered.connect(self.emitSettings)
        self.action20x.triggered.connect(self.on20x)
        self.action60x.triggered.connect(self.on60x)
        self.subBackMethod.currentIndexChanged.connect(self.subtractBackgroundFunction)
        self.invertLabel.setEnabled(True)
        self.invertToggle.setEnabled(True)

        value_changed_connections = {
            self.blurValue: [self.blurSlider.setValue, self.onBlur],
            self.blurSlider: [self.blurValue.setValue],
            self.embossValue: [self.embossSlider.setValue, self.embossFunction],
            self.embossSlider: [self.embossValue.setValue],
            self.dilationValue: [self.dilationSlider.setValue, self.onDilate],
            self.dilationSlider: [self.dilationValue.setValue],
            self.frameDiffValue: [self.frameDiffSlider.setValue],
            self.frameDiffSlider: [self.frameDiffValue.setValue],
            self.frameDiffSliderMax: [self.frameDiffValueMax.setValue],
            self.frameDiffValueMax: [self.frameDiffSliderMax.setValue],
            self.subBackValue: [self.subtractBackgroundFunction],
            self.thresholdValue: [self.thresholdSlider.setValue, self.thresholdFunction],
            self.thresholdSlider: [self.thresholdValue.setValue],
            self.adaptSliderArea: [self.adaptValueArea.setValue, self.adaptFunction],
            self.adaptValueArea: [self.adaptSliderArea.setValue, self.adaptFunction],
            self.adaptSliderC: [self.adaptValueC.setValue, self.adaptFunction],
            self.adaptValueC: [self.adaptSliderC.setValue, self.adaptFunction]
        }
        
        for obj, handlers in value_changed_connections.items():
            for handler in handlers:
                obj.valueChanged.connect(handler)
            obj.valueChanged.connect(self.emitSettings)

        toggled_connections = {
            self.subBackToggle: [self.subtractBackgroundFunction],
            self.blurToggle: [self.blurValue.setEnabled, self.blurSlider.setEnabled, self.onBlur],
            self.dilationToggle: [self.dilationSlider.setEnabled, self.onDilate, self.dilationValue.setEnabled],
            self.adaptToggle: [self.adaptValueArea.setEnabled, self.adaptSliderArea.setEnabled, self.adaptValueC.setEnabled,
                               self.adaptSliderC.setEnabled, self.adaptMethod.setEnabled, self.adaptFunction],
            self.embossToggle: [self.embossValue.setEnabled, self.embossSlider.setEnabled, self.embossFunction],
            self.thresholdToggle: [self.thresholdValue.setEnabled, self.thresholdSlider.setEnabled, self.thresholdFunction,
                                  self.autoLabel.setEnabled, self.autoToggle.setEnabled],
            self.invertToggle: [self.emitSettings],
            self.autoToggle: [self.emitSettings],
            self.frameDiffToggle: [self.emitSettings]
        }
        
        for obj, handlers in toggled_connections.items():
            for handler in handlers:
                obj.toggled.connect(handler)
            obj.toggled.connect(self.emitSettings)

        
        # initialize values
        self.thresholdVal = 0
        self.embossVal = 1
        self.area_value = 21
        self.subBackModels={"MOG2":None, "KNN":None}
        self.subBackVal = self.blurVal = self.dilateVal = self.medianVal = 0
        self.pixToum = 1/0.083 # 60x
        self.scaleLength = 50
        self.barHeight = 30
        self.posX, self.posY = 5, 5
        self.divisions = 5
        self.fontScale = 1.5
        self.dim = (640, 480)
        self.scaleBarDialog = ScaleBar(self.dim[1], self.dim[0], self.pixToum)
        self.scaleBarDialog.valuesUpdated.connect(self.updateValues)
        self.onSubBack()

        self.video_formats = [
            "avi",
            "mp4",
            "mov",
            "mkv",
            "wmv",
            "flv",
            "mpeg",
            "mpg",
        ]
        self.video_files=[]
        self.load_directory=''
        self.save_directory=os.getcwd()
        self.csvThread = QThread()
        self.csvSaver = csvSaver(self.save_directory)
        self.csvSaver.moveToThread(self.csvThread)
        self.save_signal.connect(self.csvSaver.setSaveDirectory)
        self.csvThread.start()

    def emitSettings(self):
        settings = {
                    "adaptToggle": self.adaptToggle.isChecked(),
                    "adaptMethod": self.adaptMethod.currentText(),
                    "adaptArea": self.area_value,
                    "adaptValueC": self.adaptValueC.value(),
                    "autoToggle": self.autoToggle.isChecked(),
                    "invertToggle": self.invertToggle.isChecked(),
                    "thresholdToggle": self.thresholdToggle.isChecked(),
                    "thresholdVal": self.thresholdVal,
                    "embossToggle": self.embossToggle.isChecked(),
                    "embossVal": self.embossVal,
                    "blurToggle": self.blurToggle.isChecked(),
                    "blurVal": self.blurVal,
                    "dilationToggle": self.dilationToggle.isChecked(),
                    "dilateVal": self.dilateVal,
                    "subBackToggle": self.subBackToggle.isChecked(),
                    "subBackModels": self.subBackModels,
                    "subBackMethod": self.subBackMethod.currentText(),
                    "frameDiffToggle": self.frameDiffToggle.isChecked(),
                    "frameDiffValue": self.frameDiffValue.value(),
                    "frameDiffValueMax": self.frameDiffValueMax.value(),
                    "showFPS": self.actionFPS.isChecked(),
                    "showOriginal": self.actionOriginal_Frame.isChecked(),
                    "drawContours": self.actionBounding_Boxes_2.isChecked(),
                    "batchRecord": self.recordButton.isChecked()}
        
        if self.videoLoader is not None:
            self.settings_signal.emit(settings)
    
    def updateDisplay(self, frame):
        if frame is not None:
            try:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
                h, w, ch = frame.shape
                bytes_per_line = w * ch
                
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image).scaled(self.videoDisplay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.videoDisplay.setPixmap(pixmap)
            except Exception as e:
                print(f"Error updated display: {e}")
        else:
            print("Attempted to pass Nonetype as frame!", end='\r')

    def pauseDisplay(self):
        if self.videoLoader.capExists():
            self.videoLoader.videoState(True)

    def resumeDisplay(self):
        if self.videoLoader.capExists():
            self.videoLoader.videoState(False)
        
    def updateValues(self, scaleFactor, length, width, x, y, divisions, fontScale):
        self.pixToum = scaleFactor
        self.scaleLength = length
        self.barHeight = width
        self.posX, self.posY = x, y
        self.divisions = divisions
        self.fontScale = fontScale

    def onEditScaleBar(self):
        self.scaleBarDialog.setFrameSize(self.frameSize)
        self.scaleBarDialog.show()

    def clearVideos(self):
        self.video_files.clear()
        self.menuVideos.clear()
        if self.videoLoader is not None:
            self.videoLoader.stop()
        self.videoDisplay.setPixmap(self.blank)

    def saveDirectorySelect(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory", "", options=options)
        
        if directory:
            print(f"Directory selected: {directory}")
            try:
                self.save_directory = directory
                self.actionSaveDirectory.setStatusTip(directory)
                self.save_signal.emit(directory)
                if not os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)
                    return
            except Exception as e:
                print(e)

    def on20x(self):
        self.pixToum = 4.2

    def on60x(self):
        self.pixToum = 1/0.083
        
    def thresholdFunction(self, *args):
        self.thresholdVal = (self.thresholdValue.value()/100)*255

    def embossFunction(self, *args):
        self.embossVal = (self.embossValue.value()/100)+1

    def onBlur(self, *args):
        self.blurVal=nearestOdd(int(31*self.blurValue.value()/100))

    def onDilate(self, *args):
        self.dilateVal=int(self.dilationValue.value())

    def subtractBackgroundFunction(self, *args):
        self.subBackVal = self.subBackValue.value()/100
        if self.subBackValue.value() != 0:
            QTimer.singleShot(250,self.onSubBack)

    def onSubBack(self):
        history=450
        try:
            if self.subBackMethod.currentText() == "MOG2":
                self.subBackModels["MOG2"] = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=16+48*self.subBackVal, detectShadows=False)
            elif self.subBackMethod.currentText() == "KNN":
                self.subBackModels["KNN"] = cv2.createBackgroundSubtractorKNN(dist2Threshold=100+500*self.subBackVal, history=history, detectShadows=False)
        except Exception as e:
            msg=None

    def adaptFunction(self, *args):
        if self.thresholdToggle.isChecked() and self.adaptToggle.isChecked():
            self.thresholdToggle.setChecked(False)
        sender = self.sender()
        name = sender.objectName()
        if "Area" in name:
            area_value = self.adaptValueArea.value()
            if area_value % 2 == 0:
                area_value+=1
            self.area_value = area_value
            
    def openVideo(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Video Files (*.avi;*.mp4;*.mov;*.mkv;*.wmv;*.flv;*.mpeg;*.mpg)", options=options)

        if files:
            self.load_directory = os.path.dirname(files[0])
            for file in files:
                self.parseFiles(file)
        
    def exportVideo(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory", "", options=options)
        
        if directory:
            print(f"Directory selected: {directory}")
            try:
                self.output_dir = directory
                if not os.path.isdir(directory):
                    os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                print(e)
    
    def parseFiles(self, file):
        accepted = False
        file_type = file.split(".")[-1] # return file extension
        for video_type in self.video_formats:
            if file_type == video_type:
                accepted = True # return true if file is a video
                break
        if not accepted:
            msg = f'File type: ".{file_type}" not accepted. \nAllowed file types:  ".avi," ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"'
            print(msg)
            return
        else:
            for video_file in self.video_files:
                if file == video_file:
                    msg = f'File "{file}" already added!'
                    print(msg)
                    accepted = False
                    return
        if accepted:
            msg = f'File opened: {file}'
            file_name = file.split(r'/')[-1]
            print(msg)
            try:
                file_action = QAction(os.path.basename(file), self, checkable=True)
                file_action.setObjectName(file)
                file_action.triggered.connect(lambda checked, f=file: self.loadSelectedVideo(f, checked))
                if len(self.video_files) == 0 or not self.videoLoader.capExists():
                    self.checkVideoLoader(restart=True)
                    self.videoLoader.loadVideo(file)
                    file_action.setChecked(True)
                self.video_files.append(file)
                self.menuVideos.addAction(file_action)
                
            except Exception as e:
                msg = f'Failed to add file: {file} {e}'
                print(msg)        

    def checkVideoLoader(self, restart=False):
        if restart:
            if self.videoLoader is not None:
                print("videoLoader is running! Stopping...")
                self.videoLoader.stop()
            print("Restarting videoLoader...")
        elif self.videoLoader is None:
            print("videoLoader is gone :( Starting...")
        self.videoLoader = processVideos()
        self.restartButton.clicked.connect(self.videoLoader.restartVideo)
        self.videoLoader.frame_out.connect(self.updateDisplay)
        self.videoLoader.contours_out.connect(self.csvSaver.save)
        self.videoLoader.name_out.connect(self.manageVideos)
        self.settings_signal.connect(self.videoLoader.loadSettings)
        self.emitSettings()
        self.videoLoader.start()

    def manageVideos(self, name, file):
        if self.videoLoader is not None:
            self.videoLoader.stop()
        self.videoDisplay.setPixmap(self.blank)
        self.video_files.remove(file)
        for action in self.menuVideos.actions():
            object_name = action.objectName()
            if object_name == file:
                self.menuVideos.removeAction(action)
                found_action = True
            if len(self.video_files) > 0:
                if object_name == self.video_files[0]:
                    action.setChecked(True)
                    self.checkVideoLoader(restart=True)
                    self.videoLoader.loadVideo(object_name)
                    if found_action:
                        break
            else:
                self.recordButton.setChecked(False)
                print("No more videos!")
                break   
        
    def loadSelectedVideo(self, file, checked):
        other_actions=[]
        for action in self.menuVideos.actions():
            if action.objectName() == file:
                if checked:
                    self.checkVideoLoader(restart=True)
                    self.videoLoader.loadVideo(file)
                else:
                    action.setChecked(True)
                continue
            else:
                other_actions.append(action)

        for action_to_uncheck in other_actions:
            action_to_uncheck.setChecked(False)
            
    def loadDirectorySelect(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Load Directory", "", options=options)
        
        if directory:
            print(f"Directory selected: {directory}")
            try:
                if not os.path.isdir(directory):
                    print(f"Invalid directory: {directory}")
                    return
                self.load_directory = directory
                files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                for file in files:
                    self.parseFiles(file)
            except Exception as e:
                print(e)
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): # check if dragged item has file location/url
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            Qfiles = event.mimeData().urls()
            files = [Qfile.toLocalFile() for Qfile in Qfiles] # convert qt-type to file path string
            for file in files:
                self.parseFiles(file)
            event.accept()
        else:
            event.ignore()
    
    def closeEvent(self, event):
        try:
            self._running=False
            event.accept() 
            if self.videoLoader is not None:
                self.videoLoader.stop()
            if self.csvSaver is not None:
                self.csvSaver.closeThreads()
            self.csvThread.quit()
            self.csvThread.wait()
            super().closeEvent(event)
        except Exception as e:
            msg = f"Error during close event: {e}"
            print(msg)
            event.ignore()
        print('\nExited')

if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()

if __name__ == '__main__':
    window = MainWindow()
    app.setStyle('Windows')
    window.show()
    print('Running\n')
    app.exec()