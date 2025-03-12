# npm-analyzer-light
Optimized and efficient sequential multi-object tracker: run "python npm_analyzer_light.py"

![image](https://github.com/user-attachments/assets/51b87db6-274a-4170-9469-46db6c942658)
# Why?
My previous project, [https://github.com/nolan-cummins/npm-analyzer](url), leveraged multithreading to simultaneous process frame-differencing-based object tracking using OpenCV. However, the data structure utilized and the inherent overhead to processing, in real-time, hundreds to thousands of objects imposed significant delay, taking upwards of an hour to analyze 10-20, 30-second long recordings. 

The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously within a single process. This means that in CPython, only one thread can hold control of the Python interpreter at any given time. This prevented the previous repo from taking full advantage of high-core count. This new repo is designed with that in mind. Instead of simultaneously processing large swaths of data and extracting the relevant "tracked objects" sequentially, we run 3 main threads:
- The GUI thread, that hosts the buttons, video display, etc.
- The videoLoader thread, that loads the video from disk, applies filters, and send the new frame as a numpy array for minimal overhead. This thread also communicates with csvSaver and provides the raw contours for immediate, asynchronous analysis
- The csvSaver thread, that takes the contours from videoLoader and uses concurrent.futures to open new threads with separate Python interpreters, circumnavigating the GIL. Each new thread then runs an efficient, object-oriented (as all things in Python should be) algorithm for tracking the various detected contours/objects.

Previously, the program would append information about each newly added object in a top-down dictionary, meaning the more objects detected, the slower it became to retrieve and parse data, particularly since the values were unknown and irretrievable, or I should say, not indexable without the key. Now, instead, we save the detected object information into a new object, and append to a dictionary within that object with only one level: {"frame number": "OpenCV box2D"}. 

Now, instead of parsing through every single frame, we can instead retain the most relevant detected objects and instantly retrieve, with little delay, the box2D struct from our frame of choosing, with a single for loop. This reduced the processing time for a single video from 15-20 minutes, to approximately 20-30 seconds. Technically, it takes about 1 minute, as we must first run through the entire video and apply the filters from videoLoader before performing the analysis, but nevertheless, a significant improvement.

# .csv Structure
This was designed to use in conjunction with my [https://github.com/nolan-cummins/npm](url) project and experimental setup. During my experiments, I record the actual timestamp with high precision alongside various measurables, such as voltage, current, etc. As such, during data analysis, I needed to assign each frame with its actual timestamp. Therefore, the output from this repo is the following format:
- Rows = object count (arbitrary)
- Columns = frame number

For any object in a given frame we have 3 elements:
1. x change in position (pixels from previous frame), y change in position (pixels from previous frame)
2. area in pixels²
3. x center position (pixel), y center position (pixel)
