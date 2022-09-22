import os
import warnings
from pathlib import Path

import onnx
import pkg_resources
import torch
import yaml

# YOLOv5 library
from models.experimental import attempt_load

warnings.filterwarnings("ignore")


# model load
weight = './ckpt/checkpoint.pt'
size = 640
model = attempt_load(weight, map_location='cpu')
model.eval()
print(">> model load OK!")

# warming
batch_size = 1
im = torch.zeros(batch_size, 3, size, size).to('cpu')
y = model(im)



# onnx convert
torch.onnx.export(model, im, f"./ckpt/{Path(weight).stem}.onnx",
                verbose=False,
                opset_version=12,
                training= torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes=None
                )
print(">> onnx OK!")



# conda install openvino-ie4py -c intel
# pip install 'openvino-dev[pytorch,onnx]

# settings
installed_packages = pkg_resources.working_set
installed_packages_list = sorted([i.key for i in installed_packages])
if 'openvino' not in installed_packages_list:
    os.system(f'conda install openvino-ie4py -c intel') # OpenVINO install
    os.system(f"pip install 'openvino-dev[pytorch,onnx]'") # Model Optimizer install
    print(">> openvino install OK!")
else :
    print(">> openvino install Already!")

# OpenVINO convert
# os.system(f'mo --input_model ckpt/{Path(weight).stem}.onnx --input_shape "[1,3,{size},{size}]" -s 255 --reverse_input_channels')
# os.system(f"mo --input_model ckpt/{Path(weight).stem}.onnx --output_dir {weight[:-3]}_openvino --data_type FP32")
os.system(f"mo --input_model ckpt/{Path(weight).stem}.onnx --output_dir {weight[:-3]}_openvino \
    --input_shape [1,3,{size},{size}] -s 255 --reverse_input_channels")
with open(Path(f"{weight[:-3]}_openvino") / Path(weight).with_suffix('.yaml').name, 'w') as g:
    yaml.dump({'stride': int(max(model.stride)), 'names': model.names}, g)  # add metadata.yaml
print(">> openvino OK!")
