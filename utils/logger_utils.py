import torch


def log_metrics(logger, metrics, step):
    metrics = metrics_to_scalars(metrics)
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        logger.add_scalar(k, v, step)


def metrics_to_scalars(self, metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()

        if type(v) is dict:
            v = self.metrics_to_scalars(v)

        new_metrics[k] = v

    return new_metrics