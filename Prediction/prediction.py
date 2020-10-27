import random
import glob
import numpy as np
import tensorflow as tf
from Utilities.load_image import load_image_into_numpy_array
from Utilities.load_image import plot_detections

class Prediction:
    def __init__(self, new_pipeline_fpath, fine_tuned_ckpt):
        #recover our saved model
        #generally you want to put the last ckpt from training in here
        self.fine_tuned_ckpt = fine_tuned_ckpt #'fine_tuned_model/{self.model_name}/ckpt-1'  #TODO: add a function to check the latest ckpt
        configs = config_util.get_configs_from_pipeline_file(self.new_pipeline_fpath)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(
            model=detection_model)
        ckpt.restore(os.path.join(self.fine_tuned_ckpt))
    
        self.detection_model = detection_model

    @tf.function
    def detect(self,input_tensor):
    """Run detection on an input image.

    Args:
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.

    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
        """
    preprocessed_image, shapes = self.detection_model.preprocess(input_tensor)
    prediction_dict = self.detection_model.predict(preprocessed_image, shapes)
    return self.detection_model.postprocess(prediction_dict, shapes)

    def single_test(self,test_image_dir):
        #test_image_dir = '/content/drive/My Drive/HA AI Challenge 2020/Reference Models/Dataset/Dev/'

        TEST_IMAGE_PATHS = glob.glob(test_image_dir='*.jpg')
        image_path = random.choice(TEST_IMAGE_PATHS)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect(input_tensor)
        plot_detections(
            test_images_np[0],
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.uint32)
            + label_id_offset,
            detections['detection_scores'][0].numpy(),
            category_index, figsize=(15, 20))
        plt.show()

    def test(self):
        test_images_np = []
        images_path = os.listdir(test_image_dir)
        for i in range(1, 50):
        image_path = images_path[i]
        test_images_np.append(np.expand_dims(
            load_image_into_numpy_array(image_path), axis=0))
  
if __name__ == "__main__":
      Prediction('trained_model/efficientdet_d0_coco17_tpu-32/ckpt-1',
      "pretrained_model/efficientdet_d0_coco17_tpu-32/custom_ssd_efficientdet_d0_512x512_coco17_tpu-8.config"
      )
      