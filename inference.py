import argparse
import os
import pdb
import sys
import time

import cv2
import numpy as np
import torch
from openvino.runtime import Core
from tqdm import tqdm

# YOLOv5 library
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size, check_imshow,
                           check_requirements, check_suffix, colorstr,
                           increment_path, non_max_suppression, print_args,
                           save_one_box, scale_coords, set_logging,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

weight = './ckpt/agc2021_x6_last_neck-det_last.pt'
size = 640
test_path = '../samples'
iou = 0.99
file_num = len(list(os.listdir(test_path)))

def gpu_inference():
    device = "cuda"
    model = attempt_load(weight, map_location=device)
    model.eval()

    with torch.no_grad():
        mean_fps = 0
        mean_time = 0
        total_pred = []
        
        all_time = 0
        pbar = tqdm(os.listdir(test_path))
        for img_id in pbar:
            try:
                image = cv2.imread(os.path.join(test_path, img_id))
                src = image.copy()
                image = cv2.resize(image, (size, size))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
                image = image.to(device)
                start_time = time.time()
                output = model(image)
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                all_time += (end_time - start_time)
                inf_time = end_time - start_time
                
                pred=output[0].cpu().detach()
                pred = non_max_suppression(pred, conf_thres=0.1,
                                iou_thres=0.45, 
                                classes=[0,1,2,3,4], 
                                agnostic=False,
                                max_det=1000)
                pbar.set_description(f"GPU FPS : {fps:.3f}  |  TIME : {inf_time:.3f}s, ") # | PREDICT : {predicted}")
                
                mean_fps += fps
                mean_time += inf_time
                total_pred.append(pred)

            except:  # gray scale 이미지는 패스
                continue

        print(f"Total inference time: {round(all_time, 3)}s")
        return mean_fps/file_num, mean_time/file_num, total_pred

def cpu_inference():
    device = "cpu"
    model = attempt_load(weight, map_location=device)
    model.eval()

    with torch.no_grad():
        mean_fps = 0
        mean_time = 0
        total_pred = []
        
        all_time = 0
        pbar = tqdm(os.listdir(test_path))
        for img_id in pbar:
            try:
                image = cv2.imread(os.path.join(test_path, img_id))
                src = image.copy()
                image = cv2.resize(image, (size, size))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
                start_time = time.time()
                output = model(image)
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                all_time += (end_time - start_time)
                inf_time = end_time - start_time
                
                pred = non_max_suppression(output[0], conf_thres=0.1,
                                iou_thres=0.45, 
                                classes=[0,1,2,3,4], 
                                agnostic=False,
                                max_det=1000)
                pbar.set_description(f"CPU FPS : {fps:.3f}  |  TIME : {inf_time:.3f}s, ") # | PREDICT : {predicted}")
                
                mean_fps += fps
                mean_time += inf_time
                total_pred.append(pred)

            except:  # gray scale 이미지는 패스
                continue

        print(f"Total inference time: {round(all_time, 3)}s")
        return mean_fps/file_num, mean_time/file_num, total_pred

def openvino_inference():
    # Initialize inference engine core
    ie = Core()

    model = ie.read_model('./ckpt/agc2021_x6_last_neck-det_last_openvino/agc2021_x6_last_neck-det_last.xml')
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

# def result_compare(a, b):
#     count = 0
#     total = 0
#     for a_, b_ in zip(a, b):
#         if len(a_[0]) == len(b_[0]): # number of predicted objects
#             for a_box, b_box in zip(a_, b_):
#                 total += 1
#                 if len(a_box) == 0 and len(b_box)==0:
#                     count += 1
#                 else :
#                     if len(a_box) != 0 and len(b_box) != 0:
#                         if a_box[0][-1] == b_box[0][-1]:
#                             count += 1
#         else : 
#             total += max(len(a_[0]), len(b_[0]))

#     return count / total


from utils.metrics import box_iou


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, :4], detections[:, :4])
    correct_class = labels[:, 5] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def result_compare(a, b, iou):
    count = 0
    total = 0
    for idx, (pred_a, pred_b) in enumerate(zip(a, b)):
        correct = process_batch(pred_a[0], pred_b[0], torch.Tensor([iou]))
        count += torch.sum(correct).item()
        total += torch.numel(correct)

    return count / total

if __name__ == '__main__':
    g_fps, g_mean, g_pred = gpu_inference()
    c_fps, c_mean, c_pred = cpu_inference()
    o_fps, o_mean, o_pred = openvino_inference()
    
    print(f"GPU Inference Mean FPS : {g_fps:.3f}  |  Mean TIME : {g_mean:.3f}s")
    print(f"CPU Inference Mean FPS : {c_fps:.3f}  |  Mean TIME : {c_mean:.3f}s")
    print(f"Openvino Inference Mean FPS : {o_fps:.3f}  |  Mean TIME : {o_mean:.3f}s")
    print(f'Compare Results(cpu vs gpu): {result_compare(g_pred, c_pred, iou) * 100:.4f}% SAME!!!')
    print(f'Compare Results(cpu vs openvino): {result_compare(c_pred, o_pred, iou) * 100:.4f}% SAME!!!')
    # print(f'Compare Results(gpu vs openvino): {result_compare(g_pred, o_pred) * 100:.2f}% SAME!!!')
    
    
