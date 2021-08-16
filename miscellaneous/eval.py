import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import export_inference_graph

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./models/pipeline.config --trained_checkpoint_prefix ./trainedModel/model.ckpt --output_directory ./trainedModel/graph