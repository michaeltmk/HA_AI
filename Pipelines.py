import os
from dotenv import load_dotenv
from Utilities.model_config import ModelConfig
from prepare_model.custom_pipeline_file import CustomPipelineFile

class ToolsClassifiers:
    def __init__(self):
        load_dotenv(dotenv_path="selected.env")
        self.num_steps = os.getenv("num_steps")
        self.num_eval_steps = os.getenv("num_eval_steps")

        selected_model_info = ModelConfig.get_selected_model()
        self.model_name = selected_model_info["model name"]
        self.base_pipeline_fname = selected_model_info["base_pipeline_file"]
        self.batch_size = selected_model_info["batch_size"]

        self.new_pipeline_fname = "custom_"+self.base_pipeline_fname
        self.base_pipeline_fpath = f"pretrained_model/{self.model_name}/{self.base_pipeline_fname}"
        self.new_pipeline_fpath = f"pretrained_model/{self.model_name}/{self.new_pipeline_fname}"
    
    def change_pipeline_file(self):

        load_dotenv(dotenv_path="dataset.env")
        DATASET_PATH = os.getenv("DATASET_PATH")
        pbtxt_fpath = f"{DATASET_PATH}label_map.pbtxt"
        train_record_fpath = f"{DATASET_PATH}train.record"
        test_record_fpath = f"{DATASET_PATH}dev.record"
        num_classes = ModelConfig.get_num_classes(pbtxt_fpath)
        checkpoint_fpath = f"pretrained_model/{self.model_name}/checkpoint/ckpt-0"

        old_pipe=CustomPipelineFile(self.base_pipeline_fpath)
        print("Finish loading the old pipeline config file, now rewriting......")
        old_pipe.write_custom_config(self.new_pipeline_fpath,
                                    checkpoint_fpath,
                                    train_record_fpath,
                                    test_record_fpath,
                                    pbtxt_fpath,
                                    self.batch_size,
                                    self.num_steps,
                                    num_classes)
        print("Rewrite successfully!")

    def train(self):
        os.system("mkdir -p trained_model/")

        print(f"Training the model with: --pipeline_config_path={self.new_pipeline_fpath} \
        --model_dir=trained_model/{self.model_name}/ \
        --alsologtostderr \
        --num_train_steps={self.num_steps} \
        --sample_1_of_n_eval_examples=1 \
        --num_eval_steps={self.num_eval_steps}")

        os.system(f"python3 models/research/object_detection/model_main_tf2.py \
        --pipeline_config_path={self.new_pipeline_fpath} \
        --model_dir=trained_model/{self.model_name}/ \
        --alsologtostderr \
        --num_train_steps={self.num_steps} \
        --sample_1_of_n_eval_examples=1 \
        --num_eval_steps={self.num_eval_steps}")

if __name__ == "__main__":
    app = ToolsClassifiers()
    app.change_pipeline_file()
    app.train()