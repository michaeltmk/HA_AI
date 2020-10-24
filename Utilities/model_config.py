from object_detection.utils import label_map_util
import json
from dotenv import load_dotenv

class ModelConfig:
    @classmethod
    def get_num_classes(cls,pbtxt_fname):
        
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())

    @classmethod
    def get_selected_model(cls):
        load_dotenv(dotenv_path='selected.env')

        with open("models.json") as jf:
            MODELS_CONFIG = json.loads(jf)

        pretrained_checkpoint = MODELS_CONFIG[selected_model]['pretrained_checkpoint']

        # Name of the object detection model to use.
        MODEL = MODELS_CONFIG[selected_model]['model_name']

        # Name of the pipline file in tensorflow object detection API.
        base_pipeline_file = MODELS_CONFIG[selected_model]['base_pipeline_file']

        # Training batch size fits in Colabe's Tesla K80 GPU memory for selected model.
        batch_size = MODELS_CONFIG[selected_model]['batch_size']

        return {"model name": MODEL, "pretrained_checkpoint":pretrained_checkpoint,"base_pipeline_file":base_pipeline_file,"batch_size":batch_size}