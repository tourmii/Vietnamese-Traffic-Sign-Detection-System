from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tourmii/faster_rcnn_trafficsigns", filename="best_model.pth", local_dir="./checkpoints1")

