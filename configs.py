hyperparameters = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'num_configs': 1,
    'dropout': 0,
}

sweep_config = {
    "method": "bayes", # Can be 'grid', 'random', or 'bayes'
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [16, 128, 256]},
        "learning_rate": {"values": [1e-2]},
        "num_epochs": {"values": [10]},
        "dropout": {"values": [0.3]},
    },
}

run_config = {
    "project": "mlx8-week-05-audio",
    "entity": "ewanbeattie1-n-a",
    'run_type': 'sweep',  # 'sweep', 'train' or 'test
}