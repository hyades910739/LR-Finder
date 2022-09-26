import copy
import os
import tempfile
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Dict, Union

import torch
from numpy import linspace, mean
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Cacher:
    """Cache model and optimizer's state_dict as dict or in temporary directory."""

    def __init__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.obj_dic = dict()

    def get_obj(self, name: str) -> Union[torch.nn.Module, Optimizer]:
        "get object by name."
        assert name in self.obj_dic
        dic = self.obj_dic[name]
        if dic["type"] == "memory":
            return dic["value"]
        else:
            return torch.load(dic["value"])

    def cache(self, name: str, obj: Union[torch.nn.Module, Optimizer], type: str) -> None:
        "cache object with name."
        if type == "disc":
            self._to_disc(name, obj)
        elif type == "memory":
            self._in_memory(name, obj)
        else:
            raise ValueError(f"invalid type: {type}")

    def _to_disc(self, name: str, obj: Union[torch.nn.Module, Optimizer]) -> None:
        assert name not in self.obj_dic
        path = os.path.join(self._temp_dir, name)
        torch.save(path, obj)
        self.obj_dic[name] = dict(type="disc", value=path)

    def _in_memory(self, name: str, obj: Union[torch.nn.Module, Optimizer]) -> None:
        assert name not in self.obj_dic
        copyed = copy.deepcopy(obj.state_dict())
        self.obj_dic[name] = dict(type="memory", value=copyed)

    def exit(self) -> None:
        self._temp_dir.cleanup()


class LRFinder:
    """
    Find appropriate learning rate for your model.
    """

    def __init__(self, optimizer_re_init: bool = False, model_re_init: bool = False, cache: str = "disc"):
        """_summary_

        Args:
            optimizer_re_init (bool, optional): If True, reset optimizer's state_dict every time learning rate changed.
            model_re_init (bool, optional): If True, reset model's state_dict every time learning rate changed.
            cache (str, optional): 'disc' or 'memory'. Where to save model and optimizer's state dict.
        """
        assert cache in ("disc", "memory")
        self.cache = cache
        self.model_re_init = model_re_init
        self.optimizer_re_init = optimizer_re_init
        self.iter_lr_map_dict: Dict[int, float] = dict()
        self.history = dict()
        self.cacher = Cacher()

    @property
    def avg_loss_per_step(self) -> Dict[int, Dict[str, float]]:
        history_li = [(idx, self.history[idx]["lr"], self.history[idx]["loss"]) for idx in range(len(self.history))]
        result = dict()
        for step_idx, (lr, values) in enumerate(groupby(history_li, key=itemgetter(1))):
            mean_loss = mean([v[2] for v in values])
            result[step_idx] = {"lr": lr, "loss": mean_loss}
        return result

    def _set_cache(self, model: torch.nn.Module, optimizer: Optimizer) -> None:
        if self.model_re_init:
            self.cacher.cache("model", model, self.cache)
        if self.optimizer_re_init:
            self.cacher.cache("optimizer", optimizer, self.cache)

    def _re_init(self, model: torch.nn.Module, optimizer: Optimizer) -> None:
        if self.model_re_init:
            obj = self.cacher.get_obj("model")
            model.load_state_dict(obj)
        if self.optimizer_re_init:
            obj = self.cacher.get_obj("optimizer")
            optimizer.load_state_dict(obj)

    def _set_optimizer_state_dict(self, optimizer: Optimizer) -> None:
        if self.cache == "disc":
            self._temp_dir = tempfile.TemporaryDirectory()
            self._optimizer_state_dict_name = os.path.join(self._temp_dir, "optimizer_state_dict.pt")
            torch.save(optimizer, self._optimizer_state_dict_name)
        else:
            self._optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    def re_init_optimizer_state_dict(self, optimizer: Optimizer) -> Optimizer:
        """recover optimizer's state_dict to original (state when it pass to `run` function)"""
        if self.cache == "disc":
            optimizer_state_dict = torch.load(self._optimizer_state_dict_name)
        else:
            optimizer_state_dict = self._optimizer_state_dict

        optimizer.load_state_dict(optimizer_state_dict)
        return optimizer

    def build_linear_lr(
        self, lr_min: float, lr_max: float, num_step: int = 100, n_batch_per_step: int = 1
    ) -> "LRFinder":
        lr_per_steps = [float(i) for i in linspace(lr_min, lr_max, num_step)]
        iter_lr_map_dict = {
            iter_idx: lr_per_steps[iter_idx // n_batch_per_step] for iter_idx in range(int(num_step * n_batch_per_step))
        }
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.n_batch_per_step = n_batch_per_step
        self.iter_lr_map_dict = iter_lr_map_dict
        return self

    def build_exponential_lr(
        self, lr_min: float, lr_max: float, num_step: int = 100, n_batch_per_step: int = 1
    ) -> "LRFinder":
        """
        Increase learning rate exponentially
        Args:
            lr_min (float): The initial (min) learning rate.
            lr_max (float): The final (max) learning rate.
            num_step (int, optional): How many step to increase learning rate from lr_min to lr_max.
                                      The model will run (num_step * n_batch_per_step) iteration from dataloader.
            n_batch_per_step (int, optional): How many batch (iteration) should run for a learning rate step. e.g.,
                                              If n_batch_per_step=4, then model will use same lr for batch 1,2,3,4,
                                              and increase lr, then same lr for 5,6,7,8 and increase lr,
                                              until lr reach lr_max.
        """

        lr_mult = (lr_max / lr_min) ** (1.0 / (num_step - 1))
        lr_per_steps = [lr_min * (lr_max / lr_min) ** (i / (num_step - 1)) for i in range(num_step)]
        iter_lr_map_dict = {
            iter_idx: lr_per_steps[iter_idx // n_batch_per_step] for iter_idx in range(int(num_step * n_batch_per_step))
        }
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.n_batch_per_step = n_batch_per_step
        self.iter_lr_map_dict = iter_lr_map_dict
        return self

    def _assert_lr_is_build(self) -> None:
        help_msg = (
            "iter_lr_map_dict is not build, yet. Please use `build_exponential_lr` or `build_linear_lr`"
            " to build learning rate interval."
        )
        assert len(self.iter_lr_map_dict) > 0, help_msg

    def run(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
        criterion: Callable,
        device: torch.device,
    ) -> Dict:
        self._assert_lr_is_build()
        model = model.to(device)
        model.train()
        idx = 0
        # init lr:
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.iter_lr_map_dict[idx]

        self._set_cache(model, optimizer)
        losses = []

        with tqdm(total=len(self.iter_lr_map_dict), leave=True) as pbar:

            while True:
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    losses.append(float(loss.cpu().detach().numpy()))

                    if idx + 1 > len(self.iter_lr_map_dict):
                        break

                    # set lr:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.iter_lr_map_dict[idx]
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f"current lr: {self.iter_lr_map_dict[idx]:.2e}")

                    if self.optimizer_re_init and (idx % self.n_batch_per_step == 0):
                        self._re_init(model, optimizer)
                    idx += 1
                    pbar.update()

                if idx + 1 > len(self.iter_lr_map_dict):
                    break

        losses = losses[1:]
        self.history = {idx: dict(lr=self.iter_lr_map_dict[idx], loss=loss) for idx, loss in enumerate(losses)}
        self.cacher.exit()
        return self.history
