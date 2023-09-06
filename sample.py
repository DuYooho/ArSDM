import torch
from omegaconf import OmegaConf
from utils import instantiate_from_config
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader

""" global params """
BATCH_SIZE = 8
CONFIG_FILE_PATH = "configs/ArSDM_base.yaml"
CKPT_PATH = "/xxx/MICCAI_2023_ArSDM/ArSDM_base.ckpt"
RESULT_DIR = "/xxx/results"


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print(m, u)
    model.to("cuda:0")
    model.eval()
    return model


def get_model():
    config = OmegaConf.load(CONFIG_FILE_PATH)
    model = load_model_from_config(config, CKPT_PATH)
    return model


def get_data(batch_size=8):
    config = OmegaConf.load(CONFIG_FILE_PATH)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataloader = DataLoader(data.datasets["train"], batch_size=batch_size, num_workers=0, shuffle=False)

    return train_dataloader


def log_local(save_dir, images, batch_idx):
    samples_root = os.path.join(save_dir, "samples")
    mask_root = os.path.join(save_dir, "masks")
    for k in images:
        for idx, image in enumerate(images[k]):
            if k == "samples":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)
            elif k == "conditioning":
                image = image.permute(1, 2, 0)
                image = image.squeeze(-1).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)


if __name__ == "__main__":
    model = get_model()
    train_dataloader = get_data(BATCH_SIZE)

    result_root = os.path.join(RESULT_DIR, "samples")
    os.makedirs(result_root, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            for idx, batch in enumerate(train_dataloader):
                images = model.log_images(
                    batch,
                    N=BATCH_SIZE,
                    split="train",
                )
                for k in images:
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

                log_local(os.path.join(result_root, "sample_results"), images, idx)
