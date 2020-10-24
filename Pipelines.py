import os
from dotenv import load_dotenv
from Utilities.model_config import ModelConfig
from prepare_model.custom_pipeline_file import CustomPipelineFile

class ToolsClassifiers:
    def __init__(self):
        load_dotenv(dotenv_path="selected.env")
        self.model_name = os.getenv("SELECTED_MODEL")
        self.num_steps = os.getenv("num_steps")
        self.num_eval_steps = os.getenv("num_eval_steps")
        self.base_pipeline_fname = ModelConfig.get_selected_model()["base_pipeline_file"]
        self.new_pipeline_fname = "custom_"+self.base_pipeline_fname
        self.base_pipeline_fpath = "pretrained_model/"+self.model_name+"/"+self.base_pipeline_fname
        self.new_pipeline_fpath = "pretrained_model/"+self.model_name+"/"+self.new_pipeline_fname
    
    def change_pipeline_file(self):
        old_pipe=CustomPipelineFile(self.base_pipeline_fpath)
        print("Finish loading the old pipeline config file, now rewriting......")
        old_pipe.write_custom_config(self.new_pipeline_fpath)
        print("Rewrite successfully!")

    def train(self):
        os.system("mkdir -p trained_model/")

        print(f"Training the model with: --pipeline_config_path={self.pipeline_fpath} \
        --model_dir=trained_model/ \
        --alsologtostderr \
        --num_train_steps={self.num_steps} \
        --sample_1_of_n_eval_examples=1 \
        --num_eval_steps={self.num_eval_steps}")

        os.system(f"python3 models/research/object_detection/model_main_tf2.py \
        --pipeline_config_path={self.pipeline_fpath} \
        --model_dir=trained_model/ \
        --alsologtostderr \
        --num_train_steps={self.num_steps} \
        --sample_1_of_n_eval_examples=1 \
        --num_eval_steps={self.num_eval_steps}")

if __name__ == "__main__":
    app = ToolsClassifiers()
    app.change_pipeline_file()