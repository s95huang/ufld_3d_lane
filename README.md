# ufld_3d_lane
3d lane detection using ultra fast lane detection

## design
* deployed using python (mainly target ros1 using python3)
* minimum module import possible (only need cv, onnxruntime, rospy, numpy, and other ros related msgs)
* GPU accelerated (this is done by using onnxruntime-gpu)


## Code structure:

1. test_ufldv2.py: test the ufldv2 model on a mp4 video
2. test_ufldv2_ros.py: test the ufldv2 model on ros
3. ultrafastLaneDetector: contains the related preprocessing and postprocessing code for ufldv2

## Model: 
models are stored in the release file
