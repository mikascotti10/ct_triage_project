import numpy as np, cv2, torch
from torch.utils.data import Dataset

def apply_mask_and_normalize(img_np, mask_np, size_hw, mean, std):
    # img_np: float32 en [0,1] (H,W) o (H,W,1)
    if img_np.ndim == 2:
        img_np = img_np[..., None]
    if mask_np.ndim == 2:
        mask_np = mask_np[..., None]

    Ht, Wt = size_hw
    img_r  = cv2.resize(img_np,  (Wt, Ht), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mask_r = cv2.resize(mask_np, (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # binarizar {0,1}
    if mask_r.max() > 1.0:
        mask_r /= 255.0
    mask_r = (mask_r > 0.5).astype(np.float32)

    # enmascarar (zero-out)
    img_m = img_r * mask_r

    # FIX: asegurar siempre eje de canal antes de repeat
    if img_m.ndim == 2:
        img_m = img_m[..., None]

    # normalizar DESPUÉS del masking y replicar a 3 canales
    img3 = np.repeat(img_m, 3, axis=2)
    img3 = (img3 - mean) / std

    x = torch.from_numpy(img3.transpose(2, 0, 1).copy()).float()
    return x, mask_r.squeeze().astype(np.float32)

class ExternalBinaryDS(Dataset):
    def __init__(self, df, data_root, size=224, use_masks=False, roi_mode="keep",
                 mean=0.485, std=0.229, debug=False):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.size = size
        self.use_masks = use_masks
        self.roi_mode = roi_mode
        self.mean, self.std = mean, std
        self.debug = debug
        self._last_mask_numpy = None

    def __len__(self): 
        return len(self.df)

    def _load_image_windowed_scaled(self, row):
        # Devuelve float32 en [0,1] (H, W) en escala de grises
        path = row["img_path"]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found or invalid: {path}")
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # placeholder
        return img

    def _load_mask(self, row):
        mpath = row["mask_path"]
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found or invalid: {mpath}")
        return m.astype(np.float32)

    def _label_from_row(self, row):
        return int(row["label"])

    def _fname(self, i):
        return str(self.df.iloc[i]["img_path"])

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_np = self._load_image_windowed_scaled(row)

        if self.use_masks:
            mask_np = self._load_mask(row)
        else:
            mask_np = np.ones_like(img_np, dtype=np.float32)

        x, m = apply_mask_and_normalize(
            img_np, mask_np,
            size_hw=(self.size, self.size),
            mean=self.mean, std=self.std
        )

        if self.debug:
            self._last_mask_numpy = m

        y = self._label_from_row(row)
        fn = self._fname(i)
        return x, y, fn

