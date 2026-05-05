# Bounding boxes ++
import numpy as np
import os
import glob
from construct_frames import process_matfile, process_frame
from ultralytics import YOLO

# for now lets not download from onc lets just get matlab files from a folder
dirPath = os.getcwd()
file_path = os.path.join(dirPath,'Sonar Data')
bgs_write_path = os.path.join(dirPath,'Processed Frames')

def main():
    # set up model
    model = YOLO("weights/best.pt")
    
    try:
        os.makedirs(os.path.dirname(bgs_write_path), exist_ok=True)
        print("Directories created")  
    except OSError as error:
        print("Error creating directory")

    # Get that Sonar data
    sonar_files = sorted(glob.glob(os.path.join(file_path,'*.mat')))

    # Total files in sonar data
    num_files = len(sonar_files)
    if num_files == 0:
        print('No files to process')
        
    else:
        print(f'Found {num_files} file(s)')
            
    for file in sonar_files:

        didsonParams, acousticData = process_matfile(file_path, file)

        count = 1
        
        for frame in np.arange(didsonParams['numFrames']): 
            process_frame(frame, acousticData, didsonParams, bgs_write_path, count)
            count += 1
            
        model.predict(
            source = bgs_write_path,
            save=True,
            conf = 0.25
        )
        
        # TODO ARTSY PART. 

if __name__ == '__main__':
    main()