import tarfile
import requests
import json
import os

selected_model="efficientdet-d0"


def get_selected_model(selected_model):

  with open("prepare_model/models.json") as jf:
      MODELS_CONFIG = json.load(jf)

  pretrained_checkpoint = MODELS_CONFIG[selected_model]['pretrained_checkpoint']

  # Name of the object detection model to use.
  MODEL = MODELS_CONFIG[selected_model]['model_name']

  # Name of the pipline file in tensorflow object detection API.
  base_pipeline_file = MODELS_CONFIG[selected_model]['base_pipeline_file']

  # Training batch size fits in Colabe's Tesla K80 GPU memory for selected model.
  batch_size = MODELS_CONFIG[selected_model]['batch_size']

  return {"model name": MODEL, "pretrained_checkpoint":pretrained_checkpoint,"base_pipeline_file":base_pipeline_file,"batch_size":batch_size}



pretrained_checkpoint = get_selected_model(selected_model)["pretrained_checkpoint"]
download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint
os.system("mkdir -p pretrained_model/")
print("wget "+download_tar+" pretrained_model/"+pretrained_checkpoint)
os.system("wget "+download_tar+" pretrained_model/")
# r = requests.get(download_tar, allow_redirects=True)
# with open('pretrained_model/'+pretrained_checkpoint, 'wb') as file:
#   file.write(r.content)

tar = tarfile.open(pretrained_checkpoint)
tar.extractall()
tar.close()