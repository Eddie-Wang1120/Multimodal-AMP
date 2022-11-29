import models.detr.detr_onnx as detr_onnx
import models.detr.detr_torch as detr_torch
import models.detr.detr_tvm as detr_tvm


def torch_run(img_dir):
    detr_torch.run(img_dir)

def onnx_run(img_dir):
    detr_onnx.run(img_dir)

def tvm_run(img_dir):
    detr_tvm.run(img_dir, "tvm")

def tvm_autotuned_run(img_dir):
    detr_tvm.run(img_dir, "auto_tvm")