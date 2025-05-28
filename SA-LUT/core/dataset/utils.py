import math
import os
import random
import shutil

import numpy as np  # type: ignore
import torch  # type: ignore
from PIL.ImageFilter import Color3DLUT  # type: ignore
from pillow_lut import load_cube_file  # type: ignore

COLOR_CHECKER = {
    "Dark Skin": (115, 82, 68),
    "Light Skin": (194, 150, 130),
    "Blue Sky": (98, 122, 157),
    "Foliage": (87, 108, 67),
    "Blue Flower": (133, 128, 177),
    "Bluish Green": (103, 189, 170),
    "Orange": (214, 126, 44),
    "Purplish Blue": (80, 91, 166),
    "Moderate Red": (193, 90, 99),
    "Purple": (94, 60, 108),
    "Yellow Green": (157, 188, 64),
    "Orange Yellow": (224, 163, 46),
    "Blue": (56, 61, 150),
    "Green": (70, 148, 73),
    "Red": (175, 54, 60),
    "Yellow": (231, 199, 31),
    "Magenta": (187, 86, 149),
    "Cyan": (8, 133, 161),
    "White": (243, 243, 242),
    "Neutral 8": (200, 200, 200),
    "Neutral 6.5": (160, 160, 160),
    "Neutral 5": (122, 122, 121),
    "Neutral 3.5": (85, 85, 85),
    "Black": (52, 52, 52),
}


RESERVED = {
    "unseen18": [
        "2111_R709",
        "4120_vivid",
        "4130_R709",
        "4170_R709",
        "5122_R709",
        "6610_vivid",
        "7110_vivid",
        "7120_vivid",
        "8150_vivid",
        "HollywoodBlue_Night",
        "LoConNeu",
        "Musicvideo",
        "MysteryFilm",
        "Romantic Memory",
        "SelenChrome",
        "UrbanVlogger",
        "WarmDawn",
        "cinebasemodeA",
        "Musicvideo_1",
        "Musicvideo_10",
        "Musicvideo_11",
        "Musicvideo_12",
        "Musicvideo_13",
        "Musicvideo_14",
        "Musicvideo_15",
        "Musicvideo_2",
        "Musicvideo_3",
        "Musicvideo_4",
        "Musicvideo_5",
        "Musicvideo_6",
        "Musicvideo_7",
        "Musicvideo_8",
        "Musicvideo_9",
        "MysteryFilm_1",
        "MysteryFilm_10",
        "MysteryFilm_11",
        "MysteryFilm_12",
        "MysteryFilm_13",
        "MysteryFilm_14",
        "MysteryFilm_15",
        "MysteryFilm_2",
        "MysteryFilm_3",
        "MysteryFilm_4",
        "MysteryFilm_5",
        "MysteryFilm_6",
        "MysteryFilm_7",
        "MysteryFilm_8",
        "MysteryFilm_9",
        "UrbanVlogger_1",
        "UrbanVlogger_10",
        "UrbanVlogger_11",
        "UrbanVlogger_12",
        "UrbanVlogger_13",
        "UrbanVlogger_14",
        "UrbanVlogger_15",
        "UrbanVlogger_2",
        "UrbanVlogger_3",
        "UrbanVlogger_4",
        "UrbanVlogger_5",
        "UrbanVlogger_6",
        "UrbanVlogger_7",
        "UrbanVlogger_8",
        "UrbanVlogger_9",
    ]
}


TEXT_PROMPTS = {
    "red": [
        "Reddish",
        "More red",
        "Enhance red",
        "Redder",
        "Add red",
        "Intensify red",
        "Amplify red",
        "Boost red",
        "Heighten red",
        "Strengthen red",
        "Maximize red",
        "Elevate red",
        "Red accentuation",
        "Red up",
        "Enrich red",
    ],
    "orange": [
        "More orange",
        "Enhance orange",
        "Add orange",
        "Intensify orange",
        "Amplify orange",
        "Boost orange",
        "Heighten orange",
        "Strengthen orange",
        "Maximize orange",
        "Elevate orange",
        "Orange accentuation",
        "Orange up",
        "Enrich orange",
        "Orange increase",
        "Orange up",
    ],
    "yellow": [
        "More yellow",
        "Enhance yellow",
        "Add yellow",
        "Intensify yellow",
        "Amplify yellow",
        "Boost yellow",
        "Heighten yellow",
        "Strengthen yellow",
        "Maximize yellow",
        "Elevate yellow",
        "Yellow accentuation",
        "Yellow up",
        "Enrich yellow",
        "Yellow increase",
        "Yellow up",
    ],
    "green": [
        "Greenish",
        "More green",
        "Enhance green",
        "Add green",
        "Intensify green",
        "Amplify green",
        "Boost green",
        "Heighten green",
        "Strengthen green",
        "Maximize green",
        "Elevate green",
        "Green accentuation",
        "Green up",
        "Enrich green",
        "Green increase",
        "Green up",
    ],
    "aqua": [
        "More aqua",
        "Enhance aqua",
        "Add aqua",
        "Intensify aqua",
        "Amplify aqua",
        "Boost aqua",
        "Heighten aqua",
        "Strengthen aqua",
        "Maximize aqua",
        "Elevate aqua",
        "Aqua accentuation",
        "Aqua up",
        "Enrich aqua",
        "Aqua increase",
        "Aqua up",
    ],
    "blue": [
        "Bluish",
        "More blue",
        "Enhance blue",
        "Add blue",
        "Intensify blue",
        "Amplify blue",
        "Boost blue",
        "Heighten blue",
        "Strengthen blue",
        "Maximize blue",
        "Elevate blue",
        "Blue accentuation",
        "Blue up",
        "Enrich blue",
        "Blue increase",
        "Blue up",
    ],
    "purple": [
        "More purple",
        "Enhance purple",
        "Add purple",
        "Intensify purple",
        "Amplify purple",
        "Boost purple",
        "Heighten purple",
        "Strengthen purple",
        "Maximize purple",
        "Elevate purple",
        "Purple accentuation",
        "Purple up",
        "Enrich purple",
        "Purple increase",
        "Purple up",
    ],
    "magenta": [
        "More magenta",
        "Enhance magenta",
        "Add magenta",
        "Intensify magenta",
        "Amplify magenta",
        "Boost magenta",
        "Heighten magenta",
        "Strengthen magenta",
        "Maximize magenta",
        "Elevate magenta",
        "Magenta accentuation",
        "Magenta up",
        "Enrich magenta",
        "Magenta increase",
        "Magenta up",
    ],
    "increase_brightness": [
        "increase brightness",
        "Brighten",
        "Enhance brightness",
        "Boost brightness",
        "Raise brightness",
        "Intensify brightness",
        "Amplify brightness",
        "Heighten brightness",
        "Maximize brightness",
        "More brightness",
        "Brightness up",
        "Bright up",
    ],
    "decrease_brightness": [
        "decrease brightness",
        "Dim",
        "Reduce brightness",
        "Lower brightness",
        "Minimize brightness",
        "Diminish brightness",
        "Tone down brightness",
        "Weaken brightness",
        "Brightness down",
        "Less brightness",
        "Suppress brightness",
    ],
    "increase_contrast": [
        "increase contrast",
        "Boost contrast",
        "Enhance contrast",
        "Raise contrast",
        "Intensify contrast",
        "Amplify contrast",
        "Heighten contrast",
        "Maximize contrast",
        "More contrast",
        "Contrast up",
        "Increase clarity",
    ],
    "decrease_contrast": [
        "decrease contrast",
        "Lower contrast",
        "Reduce contrast",
        "Diminish contrast",
        "Tone down contrast",
        "Weaken contrast",
        "Contrast down",
        "Less contrast",
        "Soften contrast",
        "Minimize contrast",
        "Subdue contrast",
    ],
    "increase_saturation": [
        "increase saturation",
        "Boost saturation",
        "Enhance saturation",
        "Raise saturation",
        "Intensify saturation",
        "Amplify saturation",
        "Heighten saturation",
        "Maximize saturation",
        "More saturation",
        "Saturation up",
        "Increase color depth",
    ],
    "decrease_saturation": [
        "decrease saturation",
        "Lower saturation",
        "Reduce saturation",
        "Diminish saturation",
        "Tone down saturation",
        "Weaken saturation",
        "Saturation down",
        "Less saturation",
        "Desaturate",
        "Minimize saturation",
        "Subdue saturation",
    ],
}


def get_filename(file_path):
    basename = os.path.basename(file_path)
    name, _ = os.path.splitext(basename)
    return name


def read_3dlut_from_file(file_name, return_type="tensor"):
    file = open(file_name, "r")
    lines = file.readlines()
    start, end = 0, 0  # 从cube文件读取时
    for i in range(len(lines)):
        if lines[i][0].isdigit() or lines[i].startswith("-"):
            start = i
            break
    for i in range(len(lines) - 1, start, -1):
        if lines[i][0].isdigit() or lines[i].startswith("-"):
            end = i
            break
    lines = lines[start : end + 1]
    if len(lines) == 262144:
        dim = 64
    elif len(lines) == 35937:
        dim = 33
    else:
        dim = int(np.round(math.pow(len(lines), 1 / 3)))
    # print("dim = ", dim)
    buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)
    # LUT的格式是 cbgr，其中c是 rgb
    # 在lut文件中，一行中依次是rgb
    # r是最先最多变化的，b是变化最少的
    # 往里填的过程中，k是最先最多变化的，它填在最后位置
    for i in range(0, dim):  # b
        for j in range(0, dim):  # g
            for k in range(0, dim):  # r
                n = i * dim * dim + j * dim + k
                x = lines[n].split()
                buffer[0, i, j, k] = float(x[0])  # r
                buffer[1, i, j, k] = float(x[1])  # g
                buffer[2, i, j, k] = float(x[2])  # b

    if return_type in ["numpy", "np"]:
        return buffer
    elif return_type in ["tensor", "ts"]:
        return torch.from_numpy(buffer)
        # buffer = torch.zeros(3,dim,dim,dim) # 直接用torch太慢了，不如先读入np再直接转torch
    else:
        raise ValueError("return_type should be np or ts")


def lut_interpolate(lut_path_a, lut_path_b, t_range=(0, 1)):
    lut_a_tensor, lut_b_tensor = read_3dlut_from_file(lut_path_a), read_3dlut_from_file(
        lut_path_b
    )
    lut_a, lut_b = load_cube_file(lut_path_a), load_cube_file(lut_path_b)
    t = random.uniform(*t_range)
    lut_tensor = (1 - t) * lut_a_tensor + t * lut_b_tensor
    table = (1 - t) * np.array(lut_a.table) + t * np.array(lut_b.table)
    lut = Color3DLUT(lut_a.size, table.tolist())
    return lut, lut_tensor


def check_content(path, required: list):
    if os.path.exists(path) and set(required).issubset(set(os.listdir(path))):
        return True
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return False


class LUTAugment:
    def __init__(
        self,
        scale_prob=0.8,
        scale_range=(-0.1, 0.1),
        interp_prob=0.8,
        interp_range=(0, 0.2),
    ):
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.interp_prob = interp_prob
        self.interp_range = interp_range

        base_lut_path = "assets/Standard.cube"
        self.base_lut = read_3dlut_from_file(base_lut_path, return_type="np")

    def _scale(self, lut, s_range):
        residual = lut - self.base_lut
        scale_factor = random.uniform(*s_range)
        scaled = self.base_lut + scale_factor * residual
        return np.clip(scaled, 0, 1)

    def _interpolate(self, lut_a, lut_b, t_range):
        t = random.uniform(*t_range)
        return (1 - t) * lut_a + t * lut_b

    def __call__(self, lut, lut_b):
        if random.random() < self.scale_prob:
            lut = self._scale(lut, self.scale_range)
        if random.random() < self.interp_prob:
            lut = self._interpolate(lut, lut_b, self.interp_range)
        return lut


class MixedLUTAugment:
    def __init__(
        self,
        augment_prob=0.5,
        contrast_prob=0.5,
        brightness_prob=0.5,
        warmth_prob=0.5,
        saturation_prob=0.5,
        contrast_range=(0.9, 1.1),
        brightness_range=(-0.1, 0.1),
        warmth_range=(-150, 150),
        saturation_range=(0.75, 1.5),
    ):
        self.augment_prob = augment_prob
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.warmth_prob = warmth_prob
        self.warmth_range = warmth_range
        self.saturation_prob = saturation_prob
        self.saturation_range = saturation_range

    def _adjust_contrast(self, lut, contrast_factor):
        """Adjusts the contrast of a LUT by a given contrast factor."""
        adjusted_lut = (lut - 0.5) * contrast_factor + 0.5
        adjusted_lut = np.clip(adjusted_lut, 0, 1)
        return adjusted_lut

    def _adjust_brightness(self, lut, brightness_factor):
        """Adjusts the brightness of a LUT by a given brightness factor."""
        adjusted_lut = lut + brightness_factor
        adjusted_lut = np.clip(adjusted_lut, 0, 1)
        return adjusted_lut

    def _adjust_color_temperature(self, lut, temp_factor):
        """Adjusts the color temperature of a LUT by a given factor."""
        # Convert the LUT from sRGB to linear RGB
        lut_linear = np.power(lut, 2.2)

        # Calculate the temperature adjustment
        temp = temp_factor / 1000  # Normalize the temp_factor
        r_adjust = 1 + temp  # Adjust red channel
        b_adjust = 1 - temp  # Adjust blue channel

        # Apply the adjustments
        lut_linear[:, 0] *= r_adjust  # Red channel
        lut_linear[:, 2] *= b_adjust  # Blue channel

        # Ensure the values are within valid range
        lut_linear = np.clip(lut_linear, 0, 1)

        # Convert back to sRGB
        adjusted_lut = np.power(lut_linear, 1 / 2.2)

        return adjusted_lut

    def _adjust_saturation(self, lut, saturation_factor):
        """use old contrast func"""
        # Compute the luminance (grayscale) of each color
        lum = 0.2126 * lut[:, 0] + 0.7152 * lut[:, 1] + 0.0722 * lut[:, 2]
        lum = lum[:, np.newaxis]  # Reshape to (N, 1)
        # Blend each color towards its grayscale equivalent
        adjusted_lut = lum + (lut - lum) * saturation_factor
        adjusted_lut = np.clip(adjusted_lut, 0, 1)
        return adjusted_lut

    def __call__(
        self,
        lut,
        lumi_only=False,
    ):
        """
        Apply contrast, warmth, brightness, and saturation augmentation to the LUT.

        Parameters:
        - lut (numpy.ndarray): The original LUT, shape (N, 3).
        - lumi_only (bool): If True, only apply luminance adjustments.

        Returns:
        - lut (numpy.ndarray): The augmented LUT.
        """

        if not random.random() < self.augment_prob:
            return lut

        if random.random() < self.contrast_prob:
            contrast_factor = random.uniform(*self.contrast_range)
            lut = self._adjust_contrast(lut, contrast_factor)

        if random.random() < self.brightness_prob:
            brightness_factor = random.uniform(*self.brightness_range)
            lut = self._adjust_brightness(lut, brightness_factor)

        if not lumi_only and random.random() < self.warmth_prob:
            temp_factor = random.uniform(*self.warmth_range)
            lut = self._adjust_color_temperature(lut, temp_factor)

        if not lumi_only and random.random() < self.saturation_prob:
            saturation_factor = random.uniform(*self.saturation_range)
            lut = self._adjust_saturation(lut, saturation_factor)

        return lut


class TextAugment:
    def __init__(self, probs=[0.6, 0.3, 0.1], prob_split=0.75):
        self.probs = probs
        self.prob_split = prob_split

        # Ensure probabilities sum to 1
        assert abs(sum(self.probs) - 1.0) < 1e-5, "Probabilities must sum to 1"

    def __call__(self, prompts):
        # Determine number of prompts based on probabilities
        num_prompts = random.choices(range(1, len(self.probs) + 1), weights=self.probs)[
            0
        ]

        selected_prompts = random.sample(prompts, num_prompts)

        augmented_prompts = []
        for prompt in selected_prompts:
            if "," in prompt and random.random() < self.prob_split:
                # If prompt contains ',' and random chance is met, split and pick one part
                parts = prompt.split(",")
                prompt = random.choice(parts).strip()

            # Ensure the prompt ends with a period
            if not prompt.endswith("."):
                prompt += "."

            augmented_prompts.append(prompt)

        # Combine all prompts into a single string
        final_prompt = " ".join(augmented_prompts)
        return final_prompt


if __name__ == "__main__":
    # Example usage
    text_prompts = [
        "Increase brightness, Brighten the image.",
        "Add more contrast",
        "Enhance saturation, Make colors more vivid",
        "Adjust hue",
    ]

    for _ in range(10):
        augmenter = TextAugment()
        augmented_text = augmenter(text_prompts)
        print(augmented_text)
