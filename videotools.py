"""
All functions for video analysis
"""
import numpy as np
import cv2
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from scaleBarUI import Ui_Dialog as ScaleBar_Dialog

class ScaleBar(QDialog, ScaleBar_Dialog): # save position dialog box
    valuesUpdated = Signal(float, float, int, int, int, int, float)
    
    def __init__(self, frameHeight, frameWidth, pixToUm, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle("Scale Bar")
        self.pixToUm = pixToUm

        # Set slider ranges
        self.scaleFactor_slider.setRange(0, 1000)
        self.length_slider.setRange(0, int(frameHeight*self.pixToUm))
        self.width_slider.setRange(0, frameWidth)
        self.x_slider.setRange(0, frameWidth-5)
        self.y_slider.setRange(0, frameHeight-5)
        self.divisions_slider.setRange(0, 100)
        self.fontScale_slider.setRange(0, 10)

        # Connect sliders and spinboxes
        self.scaleFactor_slider.valueChanged.connect(self.scaleFactor_input.setValue)
        self.scaleFactor_input.valueChanged.connect(self.scaleFactor_slider.setValue)

        self.length_slider.valueChanged.connect(self.length_input.setValue)
        self.length_input.valueChanged.connect(self.length_slider.setValue)

        self.width_slider.valueChanged.connect(self.width_input.setValue)
        self.width_input.valueChanged.connect(self.width_slider.setValue)

        self.x_slider.valueChanged.connect(self.x_input.setValue)
        self.x_input.valueChanged.connect(self.x_slider.setValue)

        self.y_slider.valueChanged.connect(self.y_input.setValue)
        self.y_input.valueChanged.connect(self.y_slider.setValue)

        self.divisions_slider.valueChanged.connect(self.divisions_input.setValue)
        self.divisions_input.valueChanged.connect(self.divisions_slider.setValue)

        self.fontScale_slider.valueChanged.connect(self.fontScale_input.setValue)
        self.fontScale_input.valueChanged.connect(self.fontScale_slider.setValue)    

        self.scaleFactor_input.setValue(self.pixToUm)
        self.length_input.setValue(50)
        self.width_input.setValue(30)
        self.x_input.setValue(5)
        self.y_input.setValue(5)
        self.divisions_input.setValue(5)
        self.fontScale_input.setValue(1.5)

        def set_fixed_ticks(slider, num_ticks):
            range_min = slider.minimum()
            range_max = slider.maximum()
            tick_interval = (range_max - range_min) / (num_ticks - 1)
            slider.setTickInterval(int(tick_interval))
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        # Apply the fixed tick count
        set_fixed_ticks(self.scaleFactor_slider, 10)
        set_fixed_ticks(self.length_slider, 10)
        set_fixed_ticks(self.width_slider, 10)
        set_fixed_ticks(self.x_slider, 10)
        set_fixed_ticks(self.y_slider, 10)
        set_fixed_ticks(self.divisions_slider, 10)
        set_fixed_ticks(self.fontScale_slider, 10)

        
        self.scaleBarButton.accepted.connect(self.send_values)

    def setFrameSize(self, frameSize):
        height, width = frameSize
        self.length_slider.setRange(0, int(height*self.pixToUm))
        self.width_slider.setRange(0, width)
        self.x_slider.setRange(0, width-5)
        self.y_slider.setRange(0, height-5)
    
    def send_values(self):
        self.valuesUpdated.emit(
            self.scaleFactor_input.value(),
            self.length_input.value(),
            int(self.width_input.value()),
            int(self.x_input.value()),
            int(self.y_input.value()),
            int(self.divisions_input.value()),
            self.fontScale_input.value()
        )

def frameDifferencing(frame, area_min: int=25, area_max: int=100):
    boxes, boxes2D, centers = [], [], []
    
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_min or area > area_max * 3:
            continue
        box2D = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box2D)
        box = np.intp(box)
        boxes.append(box)
        boxes2D.append(box2D)
        centers.append(np.intp(box2D[0]))
        
    return boxes, centers, boxes2D

def nearestOdd(n):
    return n if n % 2 == 1 else n + 1

def textBackground(text, fontScale, thickness, pos):
    (textWidth, textHeight), baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
    rect = [(pos[0], pos[1]), (pos[0] + textWidth, pos[1] + textHeight+baseline)]
    textPos = (pos[0], pos[1] + textHeight + baseline // 2)
    
    return (rect, textPos)

def placeLabel(frame, msg, fontScale, thickness, position):
    rect, pos = textBackground(msg, fontScale, thickness, position)
    cv2.rectangle(frame, rect[0], rect[1], color=(255, 255, 255), thickness=-1)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA) 
    return frame

def getLabelPos(box):
    # Convert Box2D to vertices (four points)
    vertices = cv2.boxPoints(box)
    
    # Sort the points: first by y-coordinate (upper-most), then by x-coordinate (right-most)
    vertices = sorted(vertices, key=lambda point: (point[1], -point[0]))

    # The first point in the sorted list is the upper-most right vertex
    return tuple(np.intp(vertices[0]))

def findAngleDif(center, corner1, box2):
    dists=[]
    for corner in box2:
        dists.append(np.linalg.norm(corner1-corner))
    nearest_index=np.argmin(dists)
    nearest_point=box2[nearest_index]
    dx_n = nearest_point[0] - center[0]
    dy_n = nearest_point[1] - center[1]
    angle_n = np.degrees(np.arctan2(dy_n, dx_n))

    dx_o = corner1[0] - center[0]
    dy_o = corner1[1] - center[1]
    angle_o = np.degrees(np.arctan2(dy_o, dx_o))

    angle_diff = (angle_n - angle_o + 180) % 360 - 180
    
    return nearest_point, angle_diff    

def calculateAngle(box2D, new_corner, pixToUm, reference_angle=0):
    """
    Calculate a stable orientation angle for a box2D object based on a constant corner.

    Args:
        box2D: The box2D object (center, size, angle).
        new_corner: The consistent corner of the box2D, regardless of rotation.
        reference_angle: The reference axis angle in degrees (0 = x-axis, 90 = y-axis).

    Returns:
        orientation_angle: The stable orientation angle of the box with respect to the reference axis.
    """
    center, _, _ = box2D
    width, height = box2D[1]

    corners = np.array(cv2.boxPoints(box2D))

    distances = np.linalg.norm(corners - new_corner, axis=1)
    closest_corner = corners[np.argsort(distances)[1]]
    
    midpoint = (new_corner + closest_corner) / 2

    points = np.array([midpoint, center])
    dx = midpoint[0] - center[0]
    dy = midpoint[1] - center[1]
    angle = np.abs(np.abs((np.degrees(np.arctan2(dy, dx)) + 180) % 360 - 180) - 180)
    if angle > 90:
        angle = np.abs(angle - 180)

    return angle, midpoint, center

class Arrow:
    def __init__(self, start_point, end_point, color=(0, 255, 0), thickness=2, tipLength=0.3):
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.thickness = thickness
        self.tipLength = tipLength

    def draw(self, image):
        # Draw the arrow on the provided image
        cv2.arrowedLine(image, self.start_point, self.end_point, self.color, self.thickness, tipLength=self.tipLength)

def random_neon_color():

    primary = np.random.choice([0, 1, 2])
    
    color = [0, 0, 0]
    color[primary] = np.random.randint(200, 256)
    
    secondary_channels = [i for i in range(3) if i != primary]
    for channel in secondary_channels:
        color[channel] = np.random.randint(0, 100)
        

    return color

def getMaxThreads():
    # Get the number of available CPUs
    numCPUs = os.cpu_count()
    
    # Default maximum threads in ThreadPoolExecutor
    maxThreads = numCPUs * 5 if numCPUs else 1
    
    return maxThreads