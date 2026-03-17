import csv
import glob
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from .constants import DATASET_DICT, DATASETS, MEAN, STD
from .data_utils import resize_depth, to_float_tensor, to_tensor
from .storm_dataset import STORMDataset, STORMDatasetEval

logger = logging.getLogger("STORM")

EGOEXO_CAMERA_LIST = ["cam01", "cam02", "cam03", "cam04"]


def parse_gopro_calib(csv_path: str, camera_list: List[str] = EGOEXO_CAMERA_LIST) -> Dict[str, Any]:
    """Parse gopro_calib.csv and return per-camera calibration data.

    Returns dict mapping cam_uid -> {cam_to_world: 4x4, intrinsics: [fx,fy,cx,cy], size: [H,W]}
    """
    calib = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_uid = row["cam_uid"]
            if cam_uid not in camera_list:
                continue

            tx = float(row["tx_world_cam"])
            ty = float(row["ty_world_cam"])
            tz = float(row["tz_world_cam"])
            qx = float(row["qx_world_cam"])
            qy = float(row["qy_world_cam"])
            qz = float(row["qz_world_cam"])
            qw = float(row["qw_world_cam"])

            rot = R.from_quat([qx, qy, qz, qw])
            cam_to_world = np.eye(4)
            cam_to_world[:3, :3] = rot.as_matrix()
            cam_to_world[:3, 3] = [tx, ty, tz]

            fx = float(row["intrinsics_0"])
            fy = float(row["intrinsics_1"])
            cx = float(row["intrinsics_2"])
            cy = float(row["intrinsics_3"])

            W = int(row["image_width"])
            H = int(row["image_height"])

            calib[cam_uid] = {
                "cam_to_world": cam_to_world,
                "intrinsics": [fx, fy, cx, cy],
                "size": [H, W],
            }
    return calib


def count_frames(image_root: str, scene_name: str, ref_camera: str = "cam01") -> int:
    """Count the number of frames for a scene by listing files in the reference camera directory."""
    pattern = os.path.join(image_root, scene_name, ref_camera, "frame_*.png")
    return len(glob.glob(pattern))


def build_annotation(
    scene_id: int,
    scene_name: str,
    calib: Dict[str, Any],
    num_frames: int,
    image_root: str,
    fps: int = 30,
) -> Dict[str, Any]:
    """Build an annotation dict matching the GaussianSTORM JSON format."""
    camera_list = [cam for cam in EGOEXO_CAMERA_LIST if cam in calib]

    annotation = {
        "dataset": "egoexo",
        "scene_id": scene_id,
        "scene_name": scene_name,
        "num_timesteps": num_frames,
        "fps": fps,
        "camera_list": camera_list,
        "normalized_time": [t / fps for t in range(num_frames)],
        "normalized_intrinsics": {},
        "camera_to_world": {},
        "camera_to_ego": {},
        "original_image_size": {},
        "relative_image_path": {},
    }

    for cam in camera_list:
        c = calib[cam]
        H, W = c["size"]
        fx, fy, cx, cy = c["intrinsics"]

        annotation["normalized_intrinsics"][cam] = [fx / W, fy / H, cx / W, cy / H]
        annotation["camera_to_ego"][cam] = np.eye(4).tolist()
        # Static cameras: same pose for all timesteps
        annotation["camera_to_world"][cam] = [c["cam_to_world"].tolist()] * num_frames
        annotation["original_image_size"][cam] = [H, W]
        annotation["relative_image_path"][cam] = [
            os.path.join(scene_name, cam, f"frame_{t:06d}.png")
            for t in range(num_frames)
        ]

    return annotation


class EgoExoDataset(STORMDataset):
    """Dataset for Ego-Exo4D that loads directly from the raw directory structure.

    No preprocessing step required. Images are loaded at full resolution and resized
    on the fly by img_transformation.
    """

    def __init__(
        self,
        image_root: str,
        annotation_root: str,
        scene_names_file: str,
        target_size: Tuple[int, int] = (160, 288),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        return_context_as_target: bool = False,
        fps: int = 30,
        load_depth: bool = False,
        depth_root: Optional[str] = None,
    ):
        # Skip STORMDataset.__init__ — we build annotations from CSVs, not JSONs
        Dataset.__init__(self)

        self.image_root = image_root
        self.annotation_root = annotation_root
        self.target_size = target_size
        self.num_context_timesteps = num_context_timesteps
        self.num_target_timesteps = num_target_timesteps
        self.num_max_cams = num_max_cams
        self.timespan = timespan
        self.equispaced = equispaced
        self.return_context_as_target = return_context_as_target
        self.fps = fps

        # Not used for EgoExo, but set for compatibility with inherited __getitem__
        self.load_depth = load_depth
        self.depth_root = depth_root
        self.load_flow = False
        self.load_dynamic_mask = False
        self.load_ground_label = False
        self.skip_sky_mask = True
        self.data_root = image_root  # not used by our get_frame override

        # Read scene names
        with open(scene_names_file, "r") as f:
            scene_names = [line.strip() for line in f if line.strip()]
        if subset_indices is not None:
            scene_names = [scene_names[i] for i in subset_indices]

        # Build annotations from CSVs
        self.annotations = []
        for scene_id, scene_name in enumerate(scene_names):
            csv_path = os.path.join(
                annotation_root, "takes", scene_name, "trajectory", "gopro_calibs.csv"
            )
            calib = parse_gopro_calib(csv_path)
            num_frames = count_frames(image_root, scene_name)
            if num_frames == 0:
                logger.warning(f"Skipping scene {scene_name}: no frames found")
                continue
            annotation = build_annotation(scene_id, scene_name, calib, num_frames, image_root, fps)
            self.annotations.append(annotation)

        logger.info(f"Loaded {len(self.annotations)} EgoExo4D scenes.")

        self.num_replicas = num_replicas
        if self.num_replicas > 1:
            self.annotations *= self.num_replicas

        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
    ) -> Dict[str, Any]:
        """Load a single frame from the EgoExo4D directory structure."""
        normalized_intrinsics = scene_json["normalized_intrinsics"]
        dataset_name = scene_json["dataset"]
        cam_to_world = scene_json["camera_to_world"]

        images, camtoworlds, intrinsics, depths = [], [], [], []

        if source_frame_idx < 0:
            source_frame_idx = frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]
        world_to_canonical = np.linalg.inv(cam_to_world[ref_camera_name][source_frame_idx])

        for camera in camera_list:
            # Load image directly from the user's directory
            img_relative_path = scene_json["relative_image_path"][camera][frame_idx]
            img_path = os.path.join(self.image_root, img_relative_path)
            img = Image.open(img_path).convert("RGB")
            img = self.img_transformation(img)
            images.append(img)

            # Load pseudo-GT depth if available
            if self.load_depth and self.depth_root is not None:
                depth_relative_path = os.path.splitext(img_relative_path)[0] + ".npy"
                depth_path = os.path.join(self.depth_root, depth_relative_path)
                depth = np.load(depth_path)
                depth = torch.tensor(depth).float()
                depth = resize_depth(depth, self.target_size)
                depths.append(depth)

            # Camera-to-world (same transform chain as other datasets)
            camtoworld = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical
                @ cam_to_world[camera][frame_idx]
                @ DATASETS[dataset_name]["opencv2dataset"]
            )
            camtoworld = to_tensor(camtoworld)
            camtoworlds.append(camtoworld)

            # Intrinsics
            fx, fy, cx, cy = np.array(normalized_intrinsics[camera])
            fx = fx * self.target_size[1]
            fy = fy * self.target_size[0]
            cx = cx * self.target_size[1]
            cy = cy * self.target_size[0]
            intrinsics.append(
                torch.tensor(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ]
                ).float()
            )

        data_dict = {
            "image": torch.stack(images),
            "camtoworld": torch.stack(camtoworlds),
            "intrinsics": torch.stack(intrinsics),
            "frame_idx": frame_idx,
        }
        if len(depths) > 0:
            data_dict["depth"] = torch.stack(depths)
        return data_dict


class EgoExoDatasetEval(EgoExoDataset):
    """Evaluation dataset for Ego-Exo4D with deterministic temporal sampling."""

    def __init__(
        self,
        image_root: str,
        annotation_root: str,
        scene_names_file: str,
        target_size: Tuple[int, int] = (160, 288),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        return_context_as_target: bool = False,
        fps: int = 30,
        load_depth: bool = False,
        depth_root: Optional[str] = None,
        scene_id_list: Optional[List[int]] = None,
        eval_stride: int = 20,
    ):
        super().__init__(
            image_root=image_root,
            annotation_root=annotation_root,
            scene_names_file=scene_names_file,
            target_size=target_size,
            num_context_timesteps=num_context_timesteps,
            num_target_timesteps=num_target_timesteps,
            num_max_cams=num_max_cams,
            timespan=timespan,
            subset_indices=subset_indices,
            num_replicas=num_replicas,
            equispaced=equispaced,
            return_context_as_target=return_context_as_target,
            fps=fps,
            load_depth=load_depth,
            depth_root=depth_root,
        )
        # Build deterministic evaluation samples
        val_sample_list = []
        if scene_id_list is None:
            scene_id_list = list(range(len(self.annotations)))
        for scene_id in scene_id_list:
            num_timesteps = self.annotations[scene_id]["num_timesteps"]
            for start_id in range(0, num_timesteps, eval_stride):
                val_sample_list.append((scene_id, start_id))
        self.val_sample_list = val_sample_list

    def __len__(self) -> int:
        return len(self.val_sample_list)

    def __getitem__(self, index: int):
        return super(EgoExoDatasetEval, self).__getitem__(
            self.val_sample_list[index][0],
            self.val_sample_list[index][1],
            return_all=True,
        )
