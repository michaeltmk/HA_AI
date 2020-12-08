import cv2
import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as trans
from torchvision import transforms as T2

from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


input_data_path = sys.argv[1]
base_path = os.path.dirname(input_data_path)
predict_result_path = sys.argv[2]
counting_result_path = sys.argv[3]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tool_class_list = [
    '\"bg\"','\"Cannula\"', '\"Chopper\"', '\"Dressing Forceps\"', '\"Fixation Ring\"', '\"Handpieces\"', '\"Hook Surgical\"', '\"Iris Scissors\"', '\"Knife Scalpel Handles\"',
    '\"Micro Scissors\"', '\"Needle Holders\"', '\"Pusher\"', '\"Spatula Surgical\"', '\"Speculum\"', '\"Spoon Surgical\"', '\"Tissue Forceps\"'
]
NUM_CLASSES = len(tool_class_list)
prob_threshold = 0.1
iou_threshold = 0.5
model_size = (480,640)

def load_model(model_base_path, num_classes=NUM_CLASSES):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(os.path.join('/usr/src/resnet50_first_trial_maxacc.pth'))
    else:
        checkpoint = torch.load(os.path.join('/usr/src/resnet50_first_trial_maxacc.pth'), map_location=torch.device('cpu'))
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

def get_transform(img):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    width,height = img.size
    transforms.append(T2.ToTensor())
    #testing 
    #transforms.append(T2.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0))
    if width > height:
        if width > model_size[1]:
            temp_height = int(height/width*model_size[1])
            temp_width = model_size[1]
            transforms.append(T2.Resize((temp_height,temp_width), interpolation=2))
            width = temp_width
            height = temp_height
    else:
        if height > model_size[0]:
            temp_height = model_size[0]
            temp_width = int(width/height*model_size[0])
            transforms.append(T2.Resize((temp_height,temp_width), interpolation=2))
            width = temp_width
            height = temp_height
    addedx = int((model_size[1] - width)/2)
    addedy = int((model_size[0] - height)/2)
    transforms.append(T2.Pad((addedx,addedy), padding_mode='constant'))
    transforms.append(T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    trans = T2.Compose(transforms)
    return trans(img)

import cv2
import numpy as np

def draw_box(img, prediction,  prob_threshold, iou_threshold):
  #prob filter
  boxes = prediction[0]['boxes'].cpu().numpy().tolist() #must be a list to feed into NMS
  probs = prediction[0]['scores'].cpu().numpy().tolist()
  labels = prediction[0]['labels'].cpu().numpy().tolist()
  keep = cv2.dnn.NMSBoxes(bboxes=boxes, scores=probs, score_threshold=prob_threshold, nms_threshold=iou_threshold)
  COLORS = np.random.randint(0, 255, size=(max(labels), 3),dtype="uint8")
  img = np.array(img)

  keep = list(map(lambda x: x[0], keep))

  boxes = [boxes[idx] for idx in keep]
  probs = [probs[idx] for idx in keep]
  labels = [labels[idx] for idx in keep]

#   for idx in range(len(boxes)):
#     box = boxes[idx]
#     box = list(map(lambda x: int(x), box))
#     label = labels[idx]
#     prob = probs[idx]
#     xmin, ymin, xmax, ymax = box
#     color = [int(c) for c in COLORS[label-1]]

#     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
#     text = f"{label}: {prob}"
#     cv2.putText(img, text, (xmin, ymin - 5),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 2)
#     plt.imshow(img)

#   plt.rcParams['figure.dpi'] = 100
#   plt.show()

  return {"boxes":boxes, "probs":probs, "labels":labels}

def rescale_bbox(bboxes:list):
  #input size of testing image: 4032*3024
  #resize to model input : 640*480
  #so ratio = 640/4032 , we hard code the ratio this time
  def timesr(x):
    return int(x/640*4032)
  return list(map(timesr, bboxes))

def output_adaptor(boxes:list, probs:list, labels:list):
  boxes = list(map(rescare_bbox,boxes))
  labels = list(map(lambda x: tool_class_list[x-1],labels))
  output = []
  for idx in range(len(boxes)):
    row = [labels[idx], probs[idx]] + boxes[idx]
    output.append(row)
  return output

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
    model = load_model(model_base_path=os.path.dirname(__file__), num_classes=NUM_CLASSES)
    model.eval()
    for _index, row in df.iterrows():
        predict_result[row['ImagePath']] = []
        counting_result[row['ImagePath']] = {}
        img_path = os.path.join(base_path, row['ImagePath'])
        # Load image
        img = Image.open(img_path).convert("RGB")
        # Tr = trans.ToTensor()
        img = get_transform(img)
        # Get Prediction
        with torch.no_grad():
            prediction = model([img.to(device)])
        # Append result to result
        if prediction and 'boxes' in prediction[0]:
            key = 0
            boxes = prediction[0]['boxes'].cpu().numpy().tolist()
            scores = prediction[0]['scores'].cpu().numpy().tolist()
            labels = prediction[0]['labels'].cpu().numpy().tolist()
            # keep = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=prob_threshold, nms_threshold=iou_threshold)
            # keep = list(map(lambda x: x[0], keep))
            # boxes = [boxes[idx] for idx in keep]
            # probs = [probs[idx] for idx in keep]
            # labels = [labels[idx] for idx in keep]

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
