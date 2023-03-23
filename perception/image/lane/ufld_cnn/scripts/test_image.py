import onnxruntime
import numpy as np
import cv2
import copy


def prepare_input(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img_height, img_width, img_channels = img.shape
    

	# print(img_height, img_width, img_channels)

	# Input values should be from -1 to 1 with a size of 288 x 800 pixels
	img_input = cv2.resize(img, (800, 288)).astype(np.float32)
	# cv2.imshow("img", img_input)
	# cv2.waitKey(0)

	# Scale input pixel values to -1 to 1
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	img_input = ((img_input / 255.0 - mean) / std)
	img_input = img_input.transpose(2, 0, 1)
	img_input = img_input[np.newaxis, :, :, :]

	# print(img_input.shape)
    # (1, 3, 288, 800)
    
	return img_input.astype(np.float32)


def visualize(img, outputs, row_anchor, cls_num_per_lane):
    # draw the lanes
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            if outputs[i, j] > 0:
                cv2.circle(img, (j, row_anchor[i]), 5, (0, 255, 0), -1)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def postprocess(outputs):
    return np.argmax(outputs)

def main():
    use_tusimple = True

    if use_tusimple:
        cls_num_per_lane = 56
        # generate the row anchor for each lane as np.array     
        row_anchor = np.array([ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284])
        onnx_model = "/mnt/0c39e9c4-f324-420d-a1e9-f20a41d147a8/personal_repos/lab_code/lab_ws/src/ufld_3d_lane/perception/image/lane/ufld_cnn/data/tusimple_288x800.onnx"

    else:
        cls_num_per_lane = 18
        row_anchor = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])
        onnx_model = "/mnt/0c39e9c4-f324-420d-a1e9-f20a41d147a8/personal_repos/lab_code/lab_ws/src/ufld_3d_lane/perception/image/lane/ufld_cnn/data/culane_288x800.onnx"

    # load the model
    sess = onnxruntime.InferenceSession(onnx_model,providers=['CUDAExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # check if we are using GPU
    print("Using GPU: ", sess.get_providers())

    # load the image
    img = cv2.imread("test.jpg")
    vis_img = copy.deepcopy(img)

    # preprocess the image
    img = prepare_input(img)

    # run the model
    outputs = sess.run([output_name], {input_name: img})[0]

    # print(outputs.shape)
    # (1, 101, 56, 4)
    num_points = 101
    num_anchors = 56
    num_lanes = 4


    # postprocess the outputs
    # outputs = postprocess(outputs)

    
    # visualize the outputs

if __name__ == "__main__":
    main()