from typing import Any, Dict

import wandb


class WandbStepLogger:
    current_step: int
    log_for_step: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.current_step = 0
        self.log_for_step = {}

    def log(self, items: Dict[str, Any]) -> None:
        self.log_for_step.update(items)

    def step(self, increment: int = 1) -> None:
        wandb.log({
            "current_step": self.current_step,
            **self.log_for_step
        })
        self.current_step += increment
        self.log_for_step = {}
