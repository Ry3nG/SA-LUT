import torch
from lightning.pytorch.cli import LightningCLI


def main():
    # set the precision mode
    torch.set_float32_matmul_precision("high")

    cli = LightningCLI()


if __name__ == "__main__":
    main()
