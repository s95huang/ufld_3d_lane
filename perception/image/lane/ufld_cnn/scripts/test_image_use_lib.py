import cv2
import numpy as np
from ultrafastLaneDetector.utils import LaneModelType
from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
import time

image_path = "test.jpg"
model_path = "/mnt/0c39e9c4-f324-420d-a1e9-f20a41d147a8/personal_repos/lab_code/lab_ws/src/ufld_3d_lane/perception/image/lane/ufld_cnn/data/tusimple_288x800.onnx"
# model_type = LaneModelType.UFLD_TUSIMPLE
model_type = LaneModelType.UFLD_TUSIMPLE


video_path = "highway-10364.mp4"

if __name__ == "__main__":
	# Initialize video
	cap = cv2.VideoCapture(video_path)
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Initialize lane detection model
	# print("Model Type : ", model_type.name)
	lane_detector = UltrafastLaneDetector(model_path, model_type)


	cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (800, 320))
	fps = 0
	frame_count = 0
	start = time.time()
	while cap.isOpened():
		try:
			# Read frame from the video
			ret, frame = cap.read()
		except:
			continue

		if ret:	

			# Detect the lanes
			output_img = lane_detector.AutoDrawLanes(frame)

			frame_count += 1
			if frame_count >= 30:
				end = time.time()
				fps = frame_count / (end - start)
				frame_count = 0
				start = time.time()
			cv2.putText(output_img, "FPS: %.2f" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow("Detected lanes", output_img)

		else:
			break
		vout.write(output_img)	
		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	vout.release()
	cap.release()
	cv2.destroyAllWindows()