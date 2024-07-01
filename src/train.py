from src.tms_source import create_and_train, run_experiments, training_dicts

version = "1.8.0"
file_name = f'../data/logs_loss_{version}'
results = run_experiments(
    training_dicts[version],
    create_and_train,
    save=True,
    file_name=file_name
    )
