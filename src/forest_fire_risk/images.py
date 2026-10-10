"""Torch dataset over the Sentinel-2 tiles, aligned with the tabular data by `sample_id`.

    from forest_fire_risk.data import load_track_a, to_tabular, with_images, split_by_year
    from forest_fire_risk.images import TileDataset
    tab = with_images(to_tabular(load_track_a()))
    train = TileDataset(tab[split_by_year(tab) == "train"])
    img, x_static, y = train[0]          # [3, 256, 256] float in [0, 1], [14] float, int
"""
import numpy as np

from forest_fire_risk.data import STATIC, load_tile


class TileDataset:
    """Works with torch.utils.data.DataLoader (it only needs __len__ / __getitem__).
    `transform` gets the uint8 HWC array and must return the image tensor; default scales to CHW float."""

    def __init__(self, meta, transform=None, image_dir=None):
        self.meta = meta.reset_index(drop=True)
        self.static = np.nan_to_num(self.meta[STATIC].to_numpy(np.float32))
        self.transform, self.image_dir = transform, image_dir

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        import torch
        img = load_tile(self.meta.sample_id[i], self.image_dir)
        img = self.transform(img) if self.transform else torch.from_numpy(img).permute(2, 0, 1).float() / 255
        return img, torch.from_numpy(self.static[i]), int(self.meta.label[i])
