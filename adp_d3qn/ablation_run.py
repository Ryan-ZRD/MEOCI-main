import os, shutil
from config import Config
from adp_d3qn.training import train_adp_d3qn
from model.vgg16 import MultiExitVGG16
from dataset.bdd100k_processor import get_bdd100k_dataloader

def _prep(cfg: Config):
    model = MultiExitVGG16(num_classes=cfg.num_classes, ac_min=cfg.ac_min).to(cfg.device)
    dataloader = get_bdd100k_dataloader(
        data_root=cfg.data_root, split="train",
        batch_size=cfg.batch_size, img_size=cfg.img_size, augment=cfg.augment
    )
    return model, dataloader

def run_all():
    variants = ["D3QN", "A-D3QN", "DP-D3QN", "ADP-D3QN"]
    for v in variants:
        cfg = Config()
        cfg.variant = v
        cfg.results_dir = os.path.join("results", v)
        if os.path.exists(cfg.results_dir):
            shutil.rmtree(cfg.results_dir)
        os.makedirs(cfg.results_dir, exist_ok=True)
        model, dataloader = _prep(cfg)
        train_adp_d3qn(model, dataloader, cfg)

if __name__ == "__main__":
    run_all()
