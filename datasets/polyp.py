import sys

sys.path.append("../")
from torch.utils.data import Dataset, DataLoader
import albumentations
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import cv2
from natsort import natsorted

workspace = Path("~/.workspace").expanduser().as_posix()


class PolypBase(Dataset):
    def __init__(self, config=None, name=None, size=384, **kwargs):
        self.kwargs = kwargs
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        self.name = name
        self.size = size

        self.preprocessor = albumentations.Compose(
            [
                albumentations.LongestMaxSize(max_size=self.size),
                albumentations.PadIfNeeded(
                    min_height=self.size,
                    min_width=self.size,
                    border_mode=cv2.BORDER_CONSTANT,
                    mask_value=0,
                ),
            ]
        )

        self._prepare()

    def _prepare(self):
        self.root = Path(workspace).joinpath("datasets/diffusion_datasets", self.name).as_posix()

        self.images_dir = Path(self.root).joinpath("images").as_posix()
        self.masks_dir = Path(self.root).joinpath("masks").as_posix()

        print(f"Preparing dataset {self.name}")

        self.images_list_absolute = Path(self.images_dir).rglob("*.png")

        self.images_list_absolute = [file_path.as_posix() for file_path in self.images_list_absolute]
        self.images_list_absolute = natsorted(self.images_list_absolute)

        self.masks_list_absolute = [
            Path(self.masks_dir).joinpath(f"{Path(p).stem}.png").as_posix() for p in self.images_list_absolute
        ]

    def __getitem__(self, i):
        data = {}
        image = Image.open(self.images_list_absolute[i]).convert("RGB")
        image = np.array(image).astype(np.uint8)

        mask = Image.open(self.masks_list_absolute[i]).convert("L")
        mask = np.array(mask).astype(np.uint8)

        _preprocessor = self.preprocessor(image=image, mask=mask)
        image, mask = _preprocessor["image"], _preprocessor["mask"]

        image = (image / 127.5 - 1.0).astype(np.float32)
        mask = mask / 255

        data["image"], data["segmentation"] = image, mask

        return data

    def __len__(self):
        return len(self.images_list_absolute)

    def getitem(self, i):
        return self.__getitem__(i)


if __name__ == "__main__":
    trainset = PolypBase(name="public_polyp_train", size=352, train=True, load_mask=True, load_hc=False)
    _data = trainset.getitem(0)

    data_loader = DataLoader(trainset, batch_size=1)

    for data in data_loader:
        image = data["image"]
        print(image.shape)  # B H W C
        print(data["image_name"])
        break
