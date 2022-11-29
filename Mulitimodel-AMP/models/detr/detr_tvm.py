import os
import time

import torch
from PIL import Image
from tvm import rpc
from tvm.contrib import graph_executor
from tvm.driver import tvmc

import utils.detr.utils_func as utils


def run(img_dir, pattern):

    if pattern == "tvm":
        package = tvmc.TVMCPackage(package_path="/home/wjh/disk/Inference_Engine/models/detr/Inference_model/detr-tvm.tar")
    elif pattern == "auto_tvm":
        package = tvmc.TVMCPackage(package_path="/home/wjh/disk/Inference_Engine/models/detr/Inference_model/detr-tvm-autotuned.tar")
    else:
        print("no suitable modules")
        return 
                
    session = rpc.LocalSession()
    lib = session.load_module(package.lib_path)
    dev = session.cuda()

    module = graph_executor.create(package.graph, lib, dev)
    module.load_params(package.params)

    files_list = os.listdir(img_dir)
    
    for i in range(len(files_list)):
        img = img_dir+files_list[i]
        img = Image.open(img,'r')
        try:
            img_data = utils.preprocess(img)
        except Exception:
            print(files_list[i]+"处理失败")
            continue

        Inference_time = 0

        module.set_input("input", img_data)
        start_time = time.time()
        module.run()
        Inference_time += time.time()-start_time
        num_outputs = module.get_num_outputs()
        outputs = {}
        for j in range(num_outputs):
            output_name = "output_{}".format(j)
            outputs[output_name] = module.get_output(j).numpy()

        outputs_prob = torch.tensor(outputs['output_0'])
        outputs_boxs = torch.tensor(outputs['output_1'])
        
        utils.postprocess(outputs_prob, outputs_boxs, img, pattern, files_list[i])
    
    print(pattern+("推理时长为%f" % Inference_time))    
    