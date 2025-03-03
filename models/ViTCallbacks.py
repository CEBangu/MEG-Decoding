import wandb
from transformers import TrainerCallback
from copy import deepcopy

class FoldEvalTrackingCallback(TrainerCallback):
    """This Callback allows the tracking of per-fold metrics, like loss. Because the way the optimizer works it wants all the folds to be
    part of the same run, but we want to get the results from each fold."""
    def __init__(self, fold_number):
        self.fold_number = fold_number

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            new_logs = {f"fold_{self.fold_number}/{key}": value for key, value in logs.items()}
            wandb.log(new_logs, commit=False)


class TrainMetricsCallback(TrainerCallback): #creds to sid8491 from Transformers forum
    """This callback allows you to track evaluation metrics on the train set, to make sure
    its learning. Unfortunately, it invloves doing ANOTHER forward pass over the data, rather than capturing
    the logits and doing the evaluations on the fly. for 10k samples, i don't think this is a problem, but this is NOT
    a scallable solution"""
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset = self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy