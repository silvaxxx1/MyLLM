import wandb

def init_wandb(project, entity, run_name, config):
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config
    )

def log_metrics(metrics):
    wandb.log(metrics)
