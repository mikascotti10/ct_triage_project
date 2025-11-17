import os, csv, time, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from .dataset import ExternalBinaryDS
from .models import make_backbone          
from .utils import eval_epoch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloaders(df, data_root, img_size, batch_size,
                     use_masks, roi_mode, num_workers=2, val_frac=0.2):
    full = ExternalBinaryDS(
        df=df,
        data_root=data_root,
        size=img_size,
        use_masks=use_masks,
        roi_mode=roi_mode,
        debug=False
    )
    n = len(full)
    nv = int(n * val_frac)
    nt = n - nv

    train_ds, val_ds = random_split(
        full,
        [nt, nv],
        generator=torch.Generator().manual_seed(123)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dl, val_dl


def train_one_epoch(model, dl, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for xb, yb, _ in tqdm(dl, leave=False, desc="train"):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(dl.dataset)


def train_baseline(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.get("out_dir", "runs/baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datos (asume df ya preprocesado con columnas: img_path, mask_path, label)
    df = cfg["df"]
    train_dl, val_dl = make_dataloaders(
        df=df,
        data_root=cfg["data_root"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        use_masks=cfg["use_masks"],
        roi_mode=cfg.get("roi_mode", "keep"),
        num_workers=cfg.get("num_workers", 2),
        val_frac=cfg.get("val_frac", 0.2),
    )

    #    Backbone seleccionable por cfg["arch"]
    #    Opciones (según tu models.py): "resnet18", "resnet50",
    #    "densenet121", "efficientnet_b0".
    model = make_backbone(
        arch=cfg.get("arch", "resnet18"),   # default: resnet18
        num_classes=2,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss",
            "val_acc", "val_prec", "val_rec", "val_f1",
            "val_auc", "pos_rate_pred", "ckpt"
        ])

    best_f1, best_ckpt = -1.0, None
    epochs = cfg["epochs"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, device, optimizer, criterion)
        val_metrics = eval_epoch(model, val_dl, device, idx_normal=0)  # 0 = “Normal”
        dt = time.time() - t0

        # guardar si mejora F1
        ckpt_path = ""
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_ckpt = out_dir / f"best_f1_epoch{epoch:.0f}_f1{best_f1:.4f}.pth"
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "val": val_metrics},
                best_ckpt
            )
            ckpt_path = str(best_ckpt)

        # log
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                f"{tr_loss:.5f}",
                *(f"{val_metrics[k]:.5f}" for k in [
                    "acc", "prec", "rec", "f1", "auc", "pos_rate_pred"
                ]),
                ckpt_path,
            ])

        print(
            f"[{epoch}/{epochs}] "
            f"loss={tr_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"({dt:.1f}s)"
        )

    return {
        "best_f1": best_f1,
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "log": str(log_path),
    }
