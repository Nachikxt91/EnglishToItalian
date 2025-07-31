from pathlib import Path

def get_config():
    return {
        "batch_size": 12,
        "num_epochs": 6,
        "lr": 2e-4,
        "seq_len": 320,
        "d_model": 256, 
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = config['model_folder']  # Just use 'weights'
    model_filename = f"{config['model_basename']}*.pth"  # Look for .pth files
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])