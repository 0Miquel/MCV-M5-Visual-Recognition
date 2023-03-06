import torch.optim as optim


def get_optimizer(config, model):
    optimizer_name = config['optimizer_name']
    settings = config['settings']

    try:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), **settings)
    except AttributeError:
        raise f"Optimizer with name {optimizer_name} not found"

    return optimizer
