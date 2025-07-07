hyperparameters = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 2,
    'num_configs': 3,
}

sweep_config = {
    "method": "bayes", # Can be 'grid', 'random', or 'bayes'
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [64]},
        "learning_rate": {"values": [1e-3]},
        "epochs": {"values": [10]},
        "embedding_size": {"values": [1, 5, 24]},
        "key_query_size": {"values": [24, 64]},
        "value_size": {"values": [24, 64]},
        "num_layers": {"values": [2, 5]},
        "dropout": {"values": [0.1]},
        "num_patches": {"values": [1]},
        "use_pos_encoding": {"values": [False]},
    },
}

run_config = {
    "project": "mlx8-week-05-audio",
    "entity": "ewanbeattie1-n-a",
    'run_type': 'train',  # 'sweep', 'train' or 'test
}