# Swap the not recognized faces

This is a pipeline built on top of the SimSwap library. The main aim is to recognize the people in the videos and swap the faces of the not recognized people with ai generated faces. Currently it supports only video files. Webcam and Luxonis camera codes are also ready but never tested.

 - Download and install
 `git clone https://github.com/izzetmustu/SimSwap && conda create env -f environment.yml && conda activate ss && cd SimSwap`
 - Create databases of known and AI generated people (edit the paths inside)
 `python createDB.py`
 - Supply options into options/recognize_options.py and run the pipeline
 `python swap_not_recognized.py --use_mask`
 - For Luxonis install dependencies
 `bash luxonis_req.sh`
 - Use other files for other tasks
 `python swap_not_recognized_dataset.py --use_mask # for directory` 
 `python swap_not_recognized_webcam/luxonis.py --use_mask # for webcam/luxonis`
