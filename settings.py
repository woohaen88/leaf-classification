import os

DATA_ROOT = os.path.join(os.environ['HOME'], 'dataset',  'leaf_classification')
args = {"batch_size": 4, 
        "train_dir": f"{DATA_ROOT}/train", 
        "val_dir": f"{DATA_ROOT}/val", 
        "test_dir": f"{DATA_ROOT}/test", 
        "epochs": 30}