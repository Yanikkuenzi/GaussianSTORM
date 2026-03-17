"""Extract pseudo ground-truth depth maps for EgoExo4D using Depth Anything V2.

Usage:
    python extract_depth_egoexo.py \
        --egoexo_image_root /path/to/egoexo/images \
        --depth_output_root /path/to/egoexo/depths \
        --depth_ckpt ckpts/depth_anything_v2_vitl.pth
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from storm.dataset.constants import IMGNET_MEAN, IMGNET_STD
from third_party.depth_anything_v2.dpt import DepthAnythingV2


class ImageListDataset(torch.utils.data.Dataset):
    """Dataset that loads images from a list of file paths."""

    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def collect_image_paths(image_root, scene_names=None):
    """Collect all frame image paths under image_root."""
    if scene_names is not None:
        paths = []
        for scene in scene_names:
            paths.extend(
                sorted(glob.glob(os.path.join(image_root, scene, "cam*", "frame_*.png")))
            )
        return paths
    return sorted(glob.glob(os.path.join(image_root, "*", "cam*", "frame_*.png")))


def get_args_parser():
    parser = argparse.ArgumentParser("Extract pseudo-GT depth for EgoExo4D", add_help=False)
    parser.add_argument("--egoexo_image_root", type=str, required=True)
    parser.add_argument("--depth_output_root", type=str, default=None,
                        help="Output root for depth maps. Defaults to <image_root>_depth")
    parser.add_argument("--depth_ckpt", type=str, default="ckpts/depth_anything_v2_vitl.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--scene_names_file", type=str, default=None,
                        help="Optional text file listing scene names to process")
    return parser


@torch.no_grad()
def extract_depth(dataloader, model, image_root, depth_output_root):
    device = next(model.parameters()).device
    for samples, paths in tqdm(dataloader, desc="Extracting depth"):
        samples = samples.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            depths = model(samples)  # (B, H, W) relative depth
        depths = depths.float().cpu().numpy()
        for i, path in enumerate(paths):
            rel = os.path.relpath(path, image_root)
            tgt = os.path.join(depth_output_root, os.path.splitext(rel)[0] + ".npy")
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            np.save(tgt, depths[i].astype(np.float32))


def main(args):
    if args.depth_output_root is None:
        args.depth_output_root = args.egoexo_image_root.rstrip("/") + "_depth"

    # Collect image paths
    scene_names = None
    if args.scene_names_file is not None:
        with open(args.scene_names_file, "r") as f:
            scene_names = [line.strip() for line in f if line.strip()]
    image_paths = collect_image_paths(args.egoexo_image_root, scene_names)
    print(f"Found {len(image_paths)} images to process.")

    # Skip already-processed images
    remaining = []
    for p in image_paths:
        rel = os.path.relpath(p, args.egoexo_image_root)
        tgt = os.path.join(args.depth_output_root, os.path.splitext(rel)[0] + ".npy")
        if not os.path.exists(tgt):
            remaining.append(p)
    print(f"Skipping {len(image_paths) - len(remaining)} already-processed images.")
    image_paths = remaining

    if len(image_paths) == 0:
        print("All images already processed.")
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_transform = transforms.Compose([
        transforms.Resize([518, 518], interpolation=Image.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
    ])

    dataset = ImageListDataset(image_paths, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # Load model
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(args.depth_ckpt, map_location="cpu"))
    model = model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    extract_depth(dataloader, model, args.egoexo_image_root, args.depth_output_root)
    print(f"Depth maps saved to {args.depth_output_root}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
