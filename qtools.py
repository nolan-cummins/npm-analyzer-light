from npm_analyzer_light import *
from tracking import *
from concurrent.futures import ProcessPoolExecutor
"""
All Qt tools/classes for main
"""

threads = os.cpu_count()
executor = ProcessPoolExecutor(max_workers=threads)

def track(contours):
    t1 = time()
    tracked_objects = trackObjects(contours)
    data = extractData(tracked_objects)
    print(f"Time elapsed: {time()-t1:.2f} s")
    return data

def processData(contours, dataframe, full_filename, full_dataname):
    print(f"Saving contours for: {full_filename}")
    dataframe.to_csv(full_filename, header=True, mode='w')
    print(f"Tracking contours for: {full_filename}")
    extracted_data = track(contours)
    print(f"Saving extracted data for: {full_dataname}")
    extracted_data.to_csv(full_dataname, header=True, mode='w')
        
class csvSaver(QObject):
    def __init__(self, save_directory):
        super().__init__()
        self.save_directory = save_directory
        self.init_mutex = QMutex()

    def openCSV(self, directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[-1] == ".csv"]
        contour_data = pd.read_csv(files[0], index_col=0)
        contours = convertFromStrings(contour_data)
        return contours

    def setSaveDirectory(self, directory):
        with QMutexLocker(self.init_mutex):
            self.save_directory = directory
        
    @Slot(object, str)
    def save(self, contours, filename):
        with QMutexLocker(self.init_mutex):
            directory = self.save_directory
            os.makedirs(f"{directory}/contours", exist_ok=True)
            os.makedirs(f"{directory}/extracted data", exist_ok=True)
        if contours and len(contours) > 0:
            max_len = max(len(lst) for lst in contours.values())
            dataframe = pd.DataFrame({
                key: lst + [pd.NA] * (max_len - len(lst)) for key, lst in contours.items()
            })
            try:
                full_filename = f'{directory}/contours/{filename.split(".")[0]}_c.csv'
                full_dataname = f'{directory}/extracted data/{filename.split(".")[0]}_ed.csv'
                future = executor.submit(processData, contours, dataframe, full_filename, full_dataname)
                future.result()
            except Exception as e:
                print(f'Error saving {filename.split(".")[0]}: {e}')  

    def closeThreads(self):
        executor.shutdown(wait=True)
