import os
import time

import onnxruntime
import torch
from PIL import Image

import utils.detr.utils_func as utils


def run(img_dir):
    pattern = "onnx"

    session = onnxruntime.InferenceSession("/home/wjh/disk/Inference_Engine/models/detr/Inference_model/detr.onnx", providers=['CUDAExecutionProvider'])
    input_node_name = session.get_inputs()[0].name
    output_node_name = [node.name for node in session.get_outputs()]


    files_list = os.listdir(img_dir)

    for i in range(len(files_list)):
        img = img_dir+files_list[i]
        img = Image.open(img, 'r')
        try:
            img_data = utils.preprocess(img)
        except Exception:
            print(files_list[i]+"处理失败")
            continue

        Inference_time = 0

        start_time = time.time()
        outputs = {}
        outputs = session.run(output_names=output_node_name, 
        input_feed={input_node_name: img_data.numpy()})
        

        outputs_prob = None
        outputs_boxs = None

        j = 0
        for x in outputs:
            if j == 0: outputs_prob = torch.tensor(x)
            elif j == 1: outputs_boxs = torch.tensor(x)
            j = j+1
        
        Inference_time += time.time()-start_time
        utils.postprocess(outputs_prob, outputs_boxs, img, pattern, files_list[i])

    print(pattern+("推理时长为%f" % Inference_time))  

