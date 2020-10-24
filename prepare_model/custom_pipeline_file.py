#write custom configuration file by slotting our dataset, model checkpoint, and training parameters into the base pipeline file

import re

class CustomPipelineFile:
    def __init__(self, pipeline_path):
        with open(pipeline_path) as file:
            self.pipeline = file.read()
    def write_custom_config(self,
                            new_path,
                            checkpoint_fpath,
                            train_record_fpath,
                            test_record_fpath,
                            label_map_pbtxt_fpath,
                            batch_size,
                            num_steps,
                            num_classes
                            ):
        print('writing custom configuration file')
        s = self.pipeline
        with open(new_path, 'w') as f:
            
            # fine_tune_checkpoint
            s = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(checkpoint_fpath), s)
            
            # tfrecord files train and test.
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

            # label_map_path
            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

            # Set training batch_size.
            s = re.sub('batch_size: [0-9]+',
                    'batch_size: {}'.format(batch_size), s)

            # Set training steps, num_steps
            s = re.sub('num_steps: [0-9]+',
                    'num_steps: {}'.format(num_steps), s)
            
            # Set number of classes num_classes.
            s = re.sub('num_classes: [0-9]+',
                    'num_classes: {}'.format(num_classes), s)
            
            #fine-tune checkpoint type
            s = re.sub(
                'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
                
            f.write(s)

