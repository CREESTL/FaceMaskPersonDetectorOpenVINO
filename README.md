# FaceMaskPersonDetector
## How to run it?
- download weights for yolov3 [here](https://yadi.sk/d/WuMc8B3krEHh1Q) and put it near `yolov3.cfg`
- run `setupvars.bat`
- run `face_mask_person_detector.py` via <strong>command line</strong>
____
If you want to test it on a different video - not webcam, change line `328` to `cap = cv.VideoCapture("test_imgs/doctor.mp4")`

This program can:
* detect&count persons on the video
* detect person's face
* detect a medical mask (yolov3 was trained to do this from scratch)
* check if a mask is on person's face
