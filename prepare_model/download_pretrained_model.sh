#!bash
pretrained_checkpoint=efficientdet_d0_coco17_tpu-32.tar.gz
base_pipeline_file=ssd_efficientdet_d0_512x512_coco17_tpu-8.config
folder_name=$(echo $pretrained_checkpoint | cut -d . -f 1)
echo $folder_name

mkdir -p pretrained_model/
cd pretrained_model/

download_tar='http://download.tensorflow.org/models/object_detection/tf2/20200711/'$pretrained_checkpoint
wget $download_tar
tar -xzf $pretrained_checkpoint
rm -rf $pretrained_checkpoint


download_config='https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/'$base_pipeline_file
wget $download_config
mv $base_pipeline_file $folder_name/$base_pipeline_file