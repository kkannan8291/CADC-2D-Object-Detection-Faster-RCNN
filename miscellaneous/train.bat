python object_detection/legacy/train.py  --train_dir=C:/AdverseWeatherDataset/CADC/TensorFlow/models/ --pipeline_config_path=C:/AdverseWeatherDataset/CADC/TensorFlow/models/research/object_detection/samples/configs/faster_rcnn_resnet101_kitti.config

python object_detection/model_main_tf2.py --model_dir=C:/AdverseWeatherDataset/CADC/TensorFlow/models/ --num_train_steps=100 --sample_1_of_n_eval_examples=1 --pipeline_config_path=C:/AdverseWeatherDataset/CADC/TensorFlow/models/research/object_detection/samples/configs/faster_rcnn_resnet101_kitti.config --alsologtostderr

python object_detection/model_main_tf2.py --model_dir=C:/AdverseWeatherDataset/CADC/TensorFlow/models/ --num_train_steps=100 --sample_1_of_n_eval_examples=1 --pipeline_config_path="C:/AdverseWeatherDataset/CADC/TensorFlow/models/TrainedModel/pipeline.config" --alsologtostderr

python model_main_tf2.py --model_dir=/models/ --num_train_steps=10000 --sample_1_of_n_eval_examples=1 --pipeline_config_path="models/pipeline.config" --alsologtostderr

python model_main_tf2.py --model_dir=/models/ --num_train_steps=100 --sample_1_of_n_eval_examples=1 --pipeline_config_path="models/faster_rcnn_resnet101_kitti.config" --alsologtostderr

python model_main_tf2.py --model_dir=/trainedModel/ --pipeline_config_path="models/pipeline.config" --alsologtostderr --checkpoint_dir=C:/AdverseWeatherDataset/Kitti/models/checkpoint

python model_main_tf2.py --model_dir=C:/AdverseWeatherDataset/Kitti/trainedModel/ --num_train_steps=1000 --pipeline_config_path="models/pipeline_CADC.config" --alsologtostderr

python model_main_tf2.py --model_dir=C:/AdverseWeatherDataset/Kitti/trainedModel/ --pipeline_config_path="models/pipeline_CADC.config" --alsologtostderr --checkpoint_dir=C:/AdverseWeatherDataset/Kitti/trainedModel/model-ckpt-cadc

# Training

python model_main_tf2.py --model_dir=C:/AdverseWeatherDataset/Kitti/trainedModel/ --num_train_steps=20000 --pipeline_config_path="C:/AdverseWeatherDataset/Kitti/models/pipeline_CADC.config" --alsologtostderr --checkpoint_every_n=500

# Evaluation

python model_main_tf2.py --model_dir=C:/AdverseWeatherDataset/Kitti/trainedModel/ --pipeline_config_path="C:/AdverseWeatherDataset/Kitti/models/pipeline_CADC.config" --alsologtostderr --checkpoint_dir=C:/AdverseWeatherDataset/Kitti/trainedModel

# Tensorboard

tensorboard --logdir=C:/AdverseWeatherDataset/Kitti/trainedModel

# Exporting a Trained model

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path "models/pipeline_CADC.config" --trained_checkpoint_dir C:/AdverseWeatherDataset/Kitti/trainedModel --output_directory .\exported-models\my_model

# Setting CUDA

set CUDA_VISIBLE_DEVICES=0