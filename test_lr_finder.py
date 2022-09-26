from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18

from lr_finder import LRFinder


@pytest.fixture(scope="session")
def load_cifir10():
    print("loading CIFAR-10")
    _ = CIFAR10("CIFAR10/", download=True, train=True)
    _ = CIFAR10("CIFAR10/", download=True, train=False)
    yield
    # tear down
    import shutil

    print("deleting CIFAR-10")
    shutil.rmtree("CIFAR10/")


def build_model() -> torch.nn.Module:
    model = resnet18(num_classes=10)
    return model


def build_dataloader(train_batch_size=64, test_batch_size=64) -> Dict[str, DataLoader]:
    """
    reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = CIFAR10("CIFAR10/", download=False, train=True, transform=train_transform)
    test = CIFAR10("CIFAR10/", download=False, train=False, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return dict(train=trainloader, test=testloader)


@pytest.mark.parametrize(
    "cache_model, cache_optimizer, cache, exponential_lr",
    [
        (True, True, "memory", True),
        (True, True, "memory", False),
        (True, True, "dist", True),
        (True, True, "dist", False),
        (True, False, "memory", True),
        (True, False, "memory", False),
        (True, False, "dist", True),
        (True, False, "dist", False),
        (False, True, "memory", True),
        (False, True, "memory", False),
        (False, True, "dist", True),
        (False, True, "dist", False),
        (False, False, "memory", True),
        (False, False, "memory", False),
        (False, False, "dist", True),
        (False, False, "dist", False),
    ],
)
def test_run_success(load_cifir10, cache_model, cache_optimizer, cache, exponential_lr):
    device = torch.device("cpu")
    model = build_model()
    model = model.to(device)
    data = build_dataloader(train_batch_size=128, test_batch_size=128)
    trainloader, testloader = data["train"], data["test"]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
    criterion = torch.nn.CrossEntropyLoss()
    # when
    lr_finder = LRFinder(optimizer_re_init=True, model_re_init=True, cache="memory")
    if exponential_lr:
        lr_finder.build_exponential_lr(1e-8, 100, num_step=4, n_batch_per_step=2).run(
            model, optimizer, trainloader, criterion, device
        )
    else:
        lr_finder.build_linear_lr(1e-8, 100, num_step=4, n_batch_per_step=2).run(
            model, optimizer, trainloader, criterion, device
        )

        assert len(lr_finder.history) > 0


def test_run_without_build_should_assert(load_cifir10):
    device = torch.device("cpu")
    model = build_model()
    model = model.to(device)
    data = build_dataloader(train_batch_size=128, test_batch_size=128)
    trainloader, testloader = data["train"], data["test"]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
    criterion = torch.nn.CrossEntropyLoss()
    # when
    lr_finder = LRFinder(optimizer_re_init=True, model_re_init=True, cache="memory")
    with pytest.raises(AssertionError):
        lr_finder.run(model, optimizer, trainloader, criterion, device)
