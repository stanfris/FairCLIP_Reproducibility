from typing import Any, Dict
import argparse

import wandb

def namespace_to_dict(namespace: argparse.Namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


class WandbLogger():
    def __init__(self, project_name: str, entity: str, config: argparse.Namespace):
        """ Initializes wandb logger and creates project """
        wandb.init(
            project=project_name,
            config=namespace_to_dict(config),
            entity=entity
        )
        self.metrics = dict()

    def finish_run(self) -> None:
        """ Finishes Wandb run """
        wandb.finish()

    def collect_metrics(self, values: Dict[str, float]) -> None:
        """ Add metrics to log """
        self.metrics.update(values)

    def upload(self) -> None:
        """ Uploads saved metrics to wandb """
        wandb.log(self.metrics)
        self.metrics = dict()


