import os
from dotenv import load_dotenv
from Utilities.model_config import ModelConfig

class ToolsClassifiers:
    def Training_pipline():
        load_dotenv(dotenv_path="selected.env")
        num_steps = os.getenv("num_steps")
        num_eval_steps = os.getenv("num_eval_steps")
        pipeline_file = ModelConfig.get_selected_model()["base_pipeline_file"]

        os.system(f"python3 models/research/object_detection/model_main_tf2.py \
        --pipeline_config_path={pipeline_file} \
        --model_dir=trained_model/ \
        --alsologtostderr \
        --num_train_steps={num_steps} \
        --sample_1_of_n_eval_examples=1 \
        --num_eval_steps={num_eval_steps}")
