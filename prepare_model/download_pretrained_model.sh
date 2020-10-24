#!bash
pretrained_checkpoint=efficientdet_d0_coco17_tpu-32.tar.gz
base_pipeline_file=ssd_efficientdet_d0_512x512_coco17_tpu-8.config



mkdir -p pretrained_model/
cd pretrained_model/

download_tar='http://download.tensorflow.org/models/object_detection/tf2/20200711/'$pretrained_checkpoint
wget $download_tar
tar -xzf $pretrained_checkpoint


download_config='https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/'$base_pipeline_file
wget $download_config