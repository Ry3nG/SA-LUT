import os
import random
from typing import List, Optional, Union

import lightning as L  # type: ignore
import torch  # type: ignore
from PIL import Image, ImageOps  # type: ignore
from PIL.ImageFilter import Color3DLUT  # type: ignore
from pillow_lut import load_cube_file  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore
from torchvision import transforms  # type: ignore
from torchvision.transforms.functional import to_tensor  # type: ignore
from torchvision.utils import save_image  # type: ignore

from core.dataset.utils import MixedLUTAugment, read_3dlut_from_file


class End2EndDataset(Dataset):
    def __init__(
        self,
        vlog_image_dir: str,
        rgb_image_dir: str,
        lut_dir: str,
        vlog_image: str,
        rgb_image: str,
        vlog_lut: str,
        rgb_lut: str,
        lut_augment: Optional[bool] = False,
        output_resolution: int = 256,
        mode: str = "train",
    ):
        super().__init__()
        self.lut_augment = lut_augment
        self.mode = mode  # train or val

        self.vlog_to_709_lut = load_cube_file("assets/Standard.cube")

        self.vlog_image_paths = self._get_image_paths(vlog_image_dir, vlog_image)
        self.rgb_image_paths = self._get_image_paths(rgb_image_dir, rgb_image)
        print(
            f"Found {len(self.vlog_image_paths)} V-Log images and {len(self.rgb_image_paths)} RGB images"
        )

        self.vlog_luts = self._get_lut_paths(os.path.join(lut_dir, "log"), vlog_lut)
        self.rgb_luts = self._get_lut_paths(os.path.join(lut_dir, "rgb"), rgb_lut)
        print(
            f"Found {len(self.vlog_luts)} V-Log LUTs and {len(self.rgb_luts)} RGB LUTs"
        )

        self.image_transform = transforms.Resize((output_resolution, output_resolution))

        if lut_augment:
            self.lut_augmentor = MixedLUTAugment(
                augment_prob=0.5,
                contrast_range=(0.95, 1.05),
                brightness_range=(-0.05, 0.05),
                warmth_range=(-50, 50),
                saturation_range=(0.9, 1.2),
            )

    def _get_image_paths(self, base_path, subfolders):
        """Fetches all image paths within explicitly defined subfolders."""
        image_paths = []
        for subfolder in subfolders:
            folder_path = os.path.join(base_path, subfolder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg", ".JPG")):
                        file_path = os.path.join(root, file)
                        image_paths.append(file_path)
        return sorted(image_paths)

    def _get_lut_paths(self, base_path, subfolders):
        """Fetches all .cube LUT paths within explicitly defined subfolders, including nested folders."""
        lut_paths = []
        for subfolder in subfolders:
            folder_path = os.path.join(base_path, subfolder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".cube"):
                        file_path = os.path.join(root, file)
                        lut_paths.append(file_path)
        return sorted(lut_paths)

    def _get_image(self, image_path, num_crops=1):
        image = Image.open(image_path).convert("RGBA").convert("RGB")
        image = ImageOps.exif_transpose(image)
        images = [self.image_transform(image) for _ in range(num_crops)]
        return images[0] if num_crops == 1 else images

    def _get_lut(self, lut_type, p_log=0.5):
        if lut_type == "random":
            lut_type = random.choices(["log", "rgb"], weights=[p_log, 1 - p_log])[0]
        if lut_type == "log":
            lut_path = random.choice(self.vlog_luts)
        elif lut_type == "rgb":
            lut_path = random.choice(self.rgb_luts)
        lut_np = read_3dlut_from_file(lut_path, return_type="np")
        lut_size = lut_np.shape[-1]
        lut_np = lut_np.transpose(1, 2, 3, 0).reshape(-1, 3)
        # augment for vlog lut
        if self.lut_augment and lut_type == "log":
            lut_np = self.lut_augmentor(lut_np)
        lut = Color3DLUT(lut_size, lut_np.tolist())
        return lut, lut_type

    def _apply_lut(self, image, lut, image_type="log", lut_type="log"):
        if image_type == lut_type:  # direct apply
            return image.filter(lut)
        elif image_type == "log" and lut_type == "rgb":
            return image.filter(self.vlog_to_709_lut).filter(lut)
        else:
            assert False, "RGB image cannot be applied with V-Log LUT"

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, index):
        if self.mode == "val":
            vlog_image_path = self.vlog_image_paths[index]
            style_image_path = self.rgb_image_paths[index]  # same index -> one-to-one

            vlog_image = self._get_image(vlog_image_path, num_crops=1)
            style_image = self._get_image(style_image_path, num_crops=1)

            vlog_tensor = to_tensor(vlog_image)
            style_tensor = to_tensor(style_image)

            return {
                "inputs": {
                    "vlog": vlog_tensor,
                    "style": style_tensor,
                },
            }

        vlog_image_path, vlog_image_path_2 = random.sample(self.vlog_image_paths, k=2)
        vlog_image = self._get_image(vlog_image_path)
        vlog_image_2 = self._get_image(vlog_image_path_2)

        # paired
        vlog_lut, lut_type = self._get_lut(lut_type="random")
        vlog_lut_neg, lut_type_neg = self._get_lut(lut_type="random")

        paired_style_image = self._apply_lut(vlog_image, vlog_lut, lut_type=lut_type)
        paired_style_image_pos = self._apply_lut(
            vlog_image_2, vlog_lut, lut_type=lut_type
        )
        paired_style_image_neg = self._apply_lut(
            vlog_image, vlog_lut_neg, lut_type=lut_type_neg
        )

        vlog_tensor = to_tensor(vlog_image)
        paired_style_tensor = to_tensor(paired_style_image)
        paired_style_tensor_pos = to_tensor(paired_style_image_pos)
        paired_style_tensor_neg = to_tensor(paired_style_image_neg)

        # unpaired
        rgb_lut1, _ = self._get_lut(lut_type="rgb")
        rgb_lut2, _ = self._get_lut(lut_type="rgb")  # Get second LUT for negation
        rgb_image_path = self.rgb_image_paths[index]
        unpaired_style_image, unpaired_style_image_pos = self._get_image(
            rgb_image_path, num_crops=2
        )

        # Apply LUT1 to both crops for style/style_pos
        unpaired_style_image = self._apply_lut(
            unpaired_style_image, rgb_lut1, image_type="log", lut_type="rgb"
        )
        unpaired_style_image_pos = self._apply_lut(
            unpaired_style_image_pos, rgb_lut1, image_type="log", lut_type="rgb"
        )
        # Apply different LUT2 to first crop for negation
        unpaired_style_image_neg = self._apply_lut(
            unpaired_style_image, rgb_lut2, image_type="log", lut_type="rgb"
        )

        unpaired_style_tensor = to_tensor(unpaired_style_image)
        unpaired_style_tensor_pos = to_tensor(unpaired_style_image_pos)
        unpaired_style_tensor_neg = to_tensor(unpaired_style_image_neg)

        # Add other style image for style transfer
        other_style_image = self._get_image(
            random.choice(self.rgb_image_paths), num_crops=1
        )
        other_style_tensor = to_tensor(other_style_image)

        return {
            "inputs": {
                "paired": {
                    "vlog": vlog_tensor,
                    "style": paired_style_tensor,
                    "style_pos": paired_style_tensor_pos,  # same style, different content
                    "style_neg": paired_style_tensor_neg,  # same content, different style
                },
                "unpaired": {
                    "vlog": vlog_tensor,
                    "style": unpaired_style_tensor,
                    "style_pos": unpaired_style_tensor_pos,  # same style, different content (crop)
                    "style_neg": unpaired_style_tensor_neg,  # same content, different style
                    "other_style": other_style_tensor,
                },
            },
        }


class End2EndDataModule(L.LightningDataModule):
    def __init__(
        self,
        vlog_image_dir: str,
        rgb_image_dir: str,
        lut_dir: str,
        vlog_image: List[str],
        rgb_image: List[str],
        vlog_lut: List[str],
        rgb_lut: List[str],
        val_vlog_image: List[str],
        val_rgb_image: List[str],
        val_vlog_lut: List[str],
        val_rgb_lut: List[str],
        test_dir: Optional[Union[str, List[str]]] = None,
        lut_augment: Optional[bool] = False,
        batch_size: Optional[int] = 8,
        output_resolution: Optional[int] = 256,
        fast_eval: Optional[bool] = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_train = End2EndDataset(
                vlog_image_dir=self.hparams["vlog_image_dir"],
                rgb_image_dir=self.hparams["rgb_image_dir"],
                lut_dir=self.hparams["lut_dir"],
                vlog_image=self.hparams["vlog_image"],
                rgb_image=self.hparams["rgb_image"],
                vlog_lut=self.hparams["vlog_lut"],
                rgb_lut=self.hparams["rgb_lut"],
                lut_augment=self.hparams["lut_augment"],
                output_resolution=self.hparams["output_resolution"],
                mode="train",
            )
        if stage in ["fit", "validate", "test"]:
            # val_dirs = self.hparams["val_dir"]
            # if not isinstance(val_dirs, list):
            #     val_dirs = [val_dirs]
            self.dataset_val = End2EndDataset(
                vlog_image_dir=self.hparams["vlog_image_dir"],
                rgb_image_dir=self.hparams["rgb_image_dir"],
                lut_dir=self.hparams["lut_dir"],
                vlog_image=self.hparams["val_vlog_image"],
                rgb_image=self.hparams["val_rgb_image"],
                vlog_lut=self.hparams["val_vlog_lut"],
                rgb_lut=self.hparams["val_rgb_lut"],
                lut_augment=False,
                output_resolution=self.hparams["output_resolution"],
                mode="val",
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, num_workers=4)


if __name__ == "__main__":

    datamodule = End2EndDataModule(
        vlog_image_dir="data/vlog_images",
        rgb_image_dir="data/rgb_images",
        lut_dir="data/lut_selected",
        vlog_image=[
            "s5m2",
            "s5m2_1016",
            "s5m2_1016_crop",
            "s5m2_1103",
            "stj",
            "test0918",
            "test1002",
            "test1023",
        ],
        rgb_image=[
            "Pexels8k",
            "ShotDeck_Screenshot",
        ],
        vlog_lut=[
            "b&w",
            "lumixlab",
            "panasonic",
            "phantom",
        ],  # 65
        rgb_lut=[
            "b&w",
            "lumixlab",
            "LutifyMe",
            "MotionArray",
        ],  # 50
        val_vlog_image=["test1002"],  # 34
        val_rgb_image=["Pexels100"],  # 100
        val_vlog_lut=["test0918"],  # 17
        val_rgb_lut=["lumixlab"],  # 50
        test_dir="data/nlut_val/panasonic",
        lut_augment=True,
        batch_size=2,
        output_resolution=256,
        fast_eval=True,
    )

    datamodule.setup("fit")
    dataset = datamodule.dataset_train
    dataloader = datamodule.train_dataloader()
    datasample = next(iter(dataloader))

    print(len(dataset))
    print({k: v.shape for k, v in datasample["inputs"]["paired"].items()})
    print({k: v.shape for k, v in datasample["inputs"]["unpaired"].items()})

    def collect_tensors(nested_dict, tensor_list=None):
        if tensor_list is None:
            tensor_list = []
        for value in nested_dict.values():
            if isinstance(
                value, dict
            ):  # If the value is another dictionary, recurse into it
                collect_tensors(value, tensor_list)
            elif isinstance(
                value, torch.Tensor
            ):  # If the value is a PyTorch tensor, add it to the list
                tensor_list.append(value)
        return tensor_list

    collect_tensors(datasample["inputs"])

    for i, batch in enumerate(dataloader):
        if i >= 16:
            break
        save_image(torch.cat(collect_tensors(batch["inputs"])), f"tmp/{i}.jpg")
