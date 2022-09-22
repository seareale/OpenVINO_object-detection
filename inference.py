import os
import time

import cv2
import numpy as np
import torch
from openvino.runtime import Core
from tqdm import tqdm

# YOLOv5 Library
from utils.general import non_max_suppression

weight = './ckpt/checkpoint.xml'
size = 640
test_path = '../samples'
iou = 0.99
file_num = len(list(os.listdir(test_path)))

def openvino_inference():
    # Initialize inference engine core
    ie = Core()

    model = ie.read_model(weight)
    compiled_model = ie.compile_model(model=model, device_name='CPU')

    all_time = 0
    
    mean_fps = 0
    mean_time = 0
    total_pred = []
    
    pbar = tqdm(os.listdir(test_path))
    for img_id in pbar:
        try:
            original_image = cv2.imread(os.path.join(test_path, img_id))
            src = original_image.copy()
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            continue

        image = original_image.copy()
        h = size
        w = size
        # _, _, h, w = compiled_model.inputs[0].shape
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        # Add N dimension to transform to NCHW
        image = np.expand_dims(image, axis=0)
        # time.sleep(1)
        start_time = time.time()
        res = compiled_model([image])[compiled_model.outputs[0]]
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        all_time += (end_time - start_time)

        # predicted = np.argmax(res, axis=1).item()
        res = torch.tensor(res)
        pred = non_max_suppression(res, conf_thres=0.1,
                                iou_thres=0.45, 
                                classes=[0,1,2,3,4], 
                                agnostic=False,
                                max_det=1000)
        inf_time = end_time - start_time
        pbar.set_description(f"OPENVINO FPS : {fps:.3f}  |  TIME : {inf_time:.3f}s, ") # | PREDICT : {predicted}")
        mean_fps += fps
        mean_time += inf_time
        total_pred.append(pred)

    print(f"Total inference time: {round(all_time, 3)}s")
    return mean_fps/file_num, mean_time/file_num, total_pred
  
  
  if __name__ == '__main__':
    o_fps, o_mean, o_pred = openvino_inference()
  
