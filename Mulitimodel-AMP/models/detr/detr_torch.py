import os
import time

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50

import utils.detr.utils_func as utils


class DETRdemo(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_decoder_layers, num_decoder_layers)
        
        self.linear_class = nn.Linear(hidden_dim, num_classes+1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        h = self.conv(x)

        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()
model = detr

def run(img_dir):
    pattern = "torch"
    files_list = os.listdir(img_dir)
    Inference_time = 0
    for i in range(len(files_list)):
        img = img_dir+files_list[i]
        img = Image.open(img,'r')
        try:
            img_data = utils.preprocess(img)
        except Exception:
            print(files_list[i]+"处理失败")
            continue
        start_time = time.time()
        outputs = model(img_data)
        Inference_time += time.time()-start_time
        outputs_prob = outputs['pred_logits']
        outputs_boxs = outputs['pred_boxes']
        utils.postprocess(outputs_prob, outputs_boxs, img, pattern, files_list[i])
    
    print(pattern+("推理时长为%f" % Inference_time))
    
    
