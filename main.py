import cv2
import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as trans

from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


input_data_path = sys.argv[1]
base_path = os.path.dirname(input_data_path)
predict_result_path = sys.argv[2]
counting_result_path = sys.argv[3]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tool_class_list = [
    '\"Cannula\"', '\"Chopper\"', '\"Dressing Forceps\"', '\"Fixation Ring\"', '\"Handpieces\"', '\"Hook Surgical\"', '\"Iris Scissors\"', '\"Knife Scalpel Handles\"',
    '\"Micro Scissors\"', '\"Needle Holders\"', '\"Pusher\"', '\"Spatula Surgical\"', '\"Speculum\"', '\"Spoon Surgical\"', '\"Tissue Forceps\"'
]

class MyEfficientNet(EfficientNet):

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

    def forward(self, inputs):
        # Modify the forward method, so that it returns only the features.
        return super().extract_features(inputs)

def load_model(model_name, model_base_path, num_classes = 2):
    if model_name == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    elif model_name == 'efficientnet-b0':
        backbone = MyEfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=num_classes)
    if model_name == 'mobilenet_v2':
        backbone.out_channels = 1280
    elif model_name == 'efficientnet-b0':
        backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(os.path.join(model_base_path, 'efficientnetb0_v1_first_trial_4.pth'))
    else:
        checkpoint = torch.load(os.path.join(model_base_path, 'efficientnetb0_v1_first_trial_4.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def export_result_file(predict_result, counting_result):
    with open(predict_result_path, 'w') as f:
        f.write("ImagePath,PredictionString (class prediction score xmin ymin xmax ymax;)\n")
        for path, values in predict_result.items():
            prediction = []
            for value in values:
                prediction.append(' '.join(str(i) for i in value))
            f.write("%s,%s\n" % (path, ';'.join(prediction)))
    with open(counting_result_path, 'w') as f:
        f.write("ImagePath,PredictionString (class count;)\n")
        for path, values in counting_result.items():
            prediction = []
            for tool_class, count in values.items():
                prediction.append("%s %s" % (tool_class, str(count)))
            f.write("%s,%s\n" % (path, ';'.join(prediction)))

if __name__ == '__main__' :
    df = pd.read_csv(input_data_path)
    # predict_result: Creating an array like 
    # {'Path': [
    #     ['ClassA', 'prediction score', 'xmin', 'ymin', 'xmax', 'ymax'], 
    #     ['ClassB', 'prediction score', 'xmin', 'ymin', 'xmax', 'ymax'],...], ...
    # }
    # counting_result: Creating an dict like 
    # {'Path': {'ClassA': 1, 'ClassB': 2}, ...}
    predict_result = {}
    counting_result = {}
    # Load Model
    model = load_model(model_name='efficientnet-b0', model_base_path=os.path.dirname(__file__), num_classes=4)
    model.eval()
    for _index, row in df.iterrows():
        predict_result[row['ImagePath']] = []
        counting_result[row['ImagePath']] = {}
        img_path = os.path.join(base_path, row['ImagePath'])
        # Load image
        img = Image.open(img_path).convert("RGB")
        Tr = trans.ToTensor()
        img = Tr(img)
        # Get Prediction
        with torch.no_grad():
            prediction = model([img.to(device)])
        # Append result to result
        if prediction and 'boxes' in prediction[0]:
            key = 0
            boxes = prediction[0]['boxes'].cpu().numpy().tolist()
            scores = prediction[0]['scores'].cpu().numpy().tolist()
            labels = prediction[0]['labels'].cpu().numpy().tolist()
            for box in boxes:
                tool_class = tool_class_list[int(labels[key])]
                score = scores[key]
                predict_result[row['ImagePath']].append([
                    tool_class, score, box[0], box[1], box[2], box[3]
                ])
                if tool_class in counting_result[row['ImagePath']]:
                    counting_result[row['ImagePath']][tool_class] += 1
                else:
                    counting_result[row['ImagePath']][tool_class] = 1
                key += 1
    # Export result to file
    export_result_file(predict_result, counting_result)
    pass
