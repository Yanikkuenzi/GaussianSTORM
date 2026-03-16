import datetime
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import storm.utils.distributed as distributed
from storm.dataset.constants import MEAN, STD
from storm.dataset.data_utils import prepare_inputs_and_targets
from storm.utils.losses import compute_scene_flow_metrics
from storm.visualization.video_maker import make_video

logger = logging.getLogger("STORM")


@torch.no_grad()
def visualize(args, model, dset_train, step, train_vis_id, device, dset_val=None, val_vis_id=None):
    model.eval()
    global_rank = distributed.get_global_rank()
    split = "train"
    for vis_id, dataset in zip([train_vis_id, val_vis_id], [dset_train, dset_val]):
        if vis_id is None or dataset is None:  # sometimes there is no validation set
            continue

        sample_id = global_rank * 80 + vis_id
        out_pth = f"{args.video_dir}/step{step}-rank{global_rank}-sample{sample_id}-{split}.mp4"

        logger.info(f"saving video to {out_pth}")
        make_video(
            dataset,
            model,
            device,
            output_filename=out_pth,
            scene_id=sample_id,
            skip_plot_gt_depth_and_flow=False,
        )

        logger.info(f"saved video to {out_pth}")
        split = "val"

    torch.cuda.empty_cache()
    return train_vis_id + 1, val_vis_id + 1 if val_vis_id is not None else None


@torch.no_grad()
def evaluate(dataloader, model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    mean = torch.tensor(MEAN).to(device)
    std = torch.tensor(STD).to(device)

    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")
    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    def get_numpy(tensor):
        return tensor.squeeze().detach().cpu().numpy()

    # Initialize running sums and counts
    total_samples, total_dynamic_samples, total_valid_dynamic_depth_samples = 0, 0, 0
    total_psnr, total_ssim, total_depth_rmse = 0.0, 0.0, 0.0
    total_occupied_psnr, total_occupied_ssim = 0.0, 0.0
    total_dynamic_psnr, total_dynamic_ssim, total_dynamic_rmse = 0.0, 0.0, 0.0
    printed = False
    pbar = tqdm(dataloader, desc="Evaluating")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device)
        input_indices = input_dict["context_frame_idx"][0].cpu().numpy().tolist()
        input_indice_start = input_indices[0]
        input_indices = [idx - input_indice_start for idx in input_indices]
        # target indices include all frames between t+0 to t+19, including the input frames
        target_indices = target_dict["target_frame_idx"][0].cpu().numpy().tolist()
        # remove the input frames from target indices to get test indices
        target_indices = [idx - input_indice_start for idx in target_indices]
        test_indices = [idx for idx in target_indices if idx not in input_indices]
        if not printed:
            logger.info(f"Input indices: {input_indices}")
            logger.info(f"Test indices: {test_indices}")
            printed = True

        has_depth = "target_depth" in target_dict
        has_sky_mask = "target_sky_masks" in target_dict
        has_dynamic_mask = "target_dynamic_masks" in target_dict

        pred_dict = model(input_dict)
        # evaluate on real target images:
        # b, t, v, c, h, w
        gt_rgb = target_dict["target_image"][:, test_indices]
        # b, t, v, h, w, c
        gt_rgb = gt_rgb.permute(0, 1, 2, 4, 5, 3) * std + mean  # transform from [-1, 1] to [0, 1]

        height, width = gt_rgb.shape[-3], gt_rgb.shape[-2]
        # btv, h, w, c
        gt_rgb = gt_rgb.reshape(-1, height, width, 3)

        if has_depth:
            gt_depth = target_dict["target_depth"][:, test_indices].view(-1, height, width)
            valid_depth_mask = gt_depth > 0.0
        if has_sky_mask:
            gt_sky_mask = target_dict["target_sky_masks"][:, test_indices].view(-1, height, width)
            occupied_mask = (gt_sky_mask == 0).bool()
        if has_dynamic_mask:
            gt_dynamic_mask = target_dict["target_dynamic_masks"][:, test_indices]
            gt_dynamic_mask = gt_dynamic_mask.view(-1, height, width)
            dynamic_mask = gt_dynamic_mask.bool()

        rendered_results = pred_dict["render_results"]
        pred_rgb = rendered_results[rendered_results["rgb_key"]][:, test_indices] * std + mean
        pred_rgb = pred_rgb.reshape(-1, height, width, 3).detach()
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        if has_depth:
            if rendered_results["decoder_depth_key"] is None:
                pred_depth = rendered_results[rendered_results["depth_key"]][:, test_indices].view(
                    -1, height, width
                )
            else:
                pred_depth = rendered_results[rendered_results["decoder_depth_key"]][
                    :, test_indices
                ].view(-1, height, width)
        psnrs, ssim_scores, depth_rmses = [], [], []
        occupied_ssims, occupied_psnrs = [], []
        dynamic_ssims, dynamic_psnrs, dynamic_depth_rmses = [], [], []
        for i in range(len(gt_rgb)):
            ssim_score = ssim(
                get_numpy(pred_rgb[i]),
                get_numpy(gt_rgb[i]),
                data_range=1.0,
                channel_axis=-1,
            )
            ssim_scores.append(ssim_score)
            psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i],
                        gt_rgb[i],
                    )
                ).item()
            )
            if has_sky_mask:
                occupied_ssims.append(
                    ssim(
                        get_numpy(pred_rgb[i]),
                        get_numpy(gt_rgb[i]),
                        data_range=1.0,
                        channel_axis=-1,
                        full=True,
                    )[1][get_numpy(occupied_mask[i])].mean()
                )
                occupied_psnrs.append(
                    -10
                    * torch.log10(
                        F.mse_loss(
                            pred_rgb[i][occupied_mask[i]],
                            gt_rgb[i][occupied_mask[i]],
                        )
                    ).item()
                )
            if has_depth:
                depth_rms = torch.sqrt(
                    F.mse_loss(
                        pred_depth[i][valid_depth_mask[i]],
                        gt_depth[i][valid_depth_mask[i]],
                    )
                ).item()
                depth_rmses.append(depth_rms)
            if has_dynamic_mask:
                if dynamic_mask[i].sum() == 0:
                    continue
                dynamic_ssims.append(
                    ssim(
                        get_numpy(pred_rgb[i]),
                        get_numpy(gt_rgb[i]),
                        data_range=1.0,
                        channel_axis=-1,
                        full=True,
                    )[1][get_numpy(dynamic_mask[i])].mean()
                )
                dynamic_psnrs.append(
                    -10
                    * torch.log10(
                        F.mse_loss(
                            pred_rgb[i][dynamic_mask[i]],
                            gt_rgb[i][dynamic_mask[i]],
                        )
                    ).item()
                )

                total_dynamic_samples += 1
                if has_depth:
                    _valid_depth_mask = dynamic_mask[i] & valid_depth_mask[i]
                    if _valid_depth_mask.sum() == 0:
                        continue
                    dynamic_depth_rms = torch.sqrt(
                        F.mse_loss(
                            pred_depth[i][_valid_depth_mask],
                            gt_depth[i][_valid_depth_mask],
                        )
                    ).item()
                    dynamic_depth_rmses.append(dynamic_depth_rms)
                    total_valid_dynamic_depth_samples += 1

        psnr_sum = np.sum(psnrs)
        ssim_sum = np.sum(ssim_scores)
        batch_size = len(gt_rgb)
        # Update running sums and counts
        total_psnr += psnr_sum
        total_ssim += ssim_sum
        total_samples += batch_size
        if has_depth:
            total_depth_rmse += np.sum(depth_rmses)
        if has_sky_mask:
            total_occupied_psnr += np.sum(occupied_psnrs)
            total_occupied_ssim += np.sum(occupied_ssims)
        if has_dynamic_mask:
            total_dynamic_psnr += np.sum(dynamic_psnrs)
            total_dynamic_ssim += np.sum(dynamic_ssims)
            total_dynamic_rmse += np.sum(dynamic_depth_rmses)
        postfix = {
            "psnr": psnr_sum / batch_size,
            "ssim": ssim_sum / batch_size,
            "avg_psnr": total_psnr / total_samples,
        }
        if has_depth:
            postfix["avg_depth_rmse"] = total_depth_rmse / total_samples
        if has_dynamic_mask and total_dynamic_samples > 0:
            postfix["avg_dynamic_psnr"] = total_dynamic_psnr / total_dynamic_samples
        pbar.set_postfix(**postfix)

    # Create tensors for sums and counts
    total_psnr_tensor = torch.tensor(total_psnr, device=device)
    total_ssim_tensor = torch.tensor(total_ssim, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)

    tensors_to_reduce = [total_psnr_tensor, total_ssim_tensor, total_samples_tensor]
    if has_depth:
        total_depth_rmse_tensor = torch.tensor(total_depth_rmse, device=device)
        tensors_to_reduce.append(total_depth_rmse_tensor)
    if has_sky_mask:
        total_occupied_psnr_tensor = torch.tensor(total_occupied_psnr, device=device)
        total_occupied_ssim_tensor = torch.tensor(total_occupied_ssim, device=device)
        tensors_to_reduce.extend([total_occupied_psnr_tensor, total_occupied_ssim_tensor])
    if has_dynamic_mask:
        total_dynamic_psnr_tensor = torch.tensor(total_dynamic_psnr, device=device)
        total_dynamic_ssim_tensor = torch.tensor(total_dynamic_ssim, device=device)
        total_dynamic_rmse_tensor = torch.tensor(total_dynamic_rmse, device=device)
        total_dynamic_samples_tensor = torch.tensor(total_dynamic_samples, device=device)
        total_valid_dynamic_depth_samples_tensor = torch.tensor(
            total_valid_dynamic_depth_samples, device=device
        )
        tensors_to_reduce.extend([
            total_dynamic_psnr_tensor, total_dynamic_ssim_tensor, total_dynamic_rmse_tensor,
            total_dynamic_samples_tensor, total_valid_dynamic_depth_samples_tensor,
        ])

    torch.cuda.synchronize()

    if distributed.is_enabled():
        for t in tensors_to_reduce:
            torch.distributed.all_reduce(t)
    result = None
    if distributed.is_main_process():
        avg_psnr = total_psnr_tensor.item() / total_samples_tensor.item()
        avg_ssim = total_ssim_tensor.item() / total_samples_tensor.item()

        result = {"psnr": avg_psnr, "ssim": avg_ssim}
        lines = [
            f"Average PSNR: {avg_psnr:.4f}",
            f"Average SSIM: {avg_ssim:.4f}",
        ]

        if has_depth:
            avg_depth_rmse = total_depth_rmse_tensor.item() / total_samples_tensor.item()
            result["depth_rmse"] = avg_depth_rmse
            lines.append(f"Average Depth RMSE: {avg_depth_rmse:.4f}")
        if has_sky_mask:
            avg_occupied_psnr = total_occupied_psnr_tensor.item() / total_samples_tensor.item()
            avg_occupied_ssim = total_occupied_ssim_tensor.item() / total_samples_tensor.item()
            result["occupied_psnr"] = avg_occupied_psnr
            result["occupied_ssim"] = avg_occupied_ssim
            lines.append(f"Average Occupied PSNR: {avg_occupied_psnr:.4f}")
            lines.append(f"Average Occupied SSIM: {avg_occupied_ssim:.4f}")
        if has_dynamic_mask and total_dynamic_samples_tensor.item() > 0:
            avg_dynamic_psnr = total_dynamic_psnr_tensor.item() / total_dynamic_samples_tensor.item()
            avg_dynamic_ssim = total_dynamic_ssim_tensor.item() / total_dynamic_samples_tensor.item()
            result["dynamic_psnr"] = avg_dynamic_psnr
            result["dynamic_ssim"] = avg_dynamic_ssim
            lines.append(f"Average Dynamic PSNR: {avg_dynamic_psnr:.4f}")
            lines.append(f"Average Dynamic SSIM: {avg_dynamic_ssim:.4f}")
            if total_valid_dynamic_depth_samples_tensor.item() > 0:
                avg_dynamic_rmse = (
                    total_dynamic_rmse_tensor.item() / total_valid_dynamic_depth_samples_tensor.item()
                )
                result["dynamic_depth_rmse"] = avg_dynamic_rmse
                lines.append(f"Average Dynamic Depth RMSE: {avg_dynamic_rmse:.4f}")

        with open(os.path.join(eval_result_dir, f"eval_{name_str}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Evaluation results saved.")
        logger.info(f"Evaluated on {total_samples_tensor.item()} samples.")
        for line in lines:
            logger.info(line)
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate_flow(dataloader, model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")
    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    (
        total_flow_epes,
        total_flow_accs_strict,
        total_flow_accs_relax,
        total_flow_angles,
        total_flow_rmse,
        total_numb_flow_samples,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pbar = tqdm(dataloader, desc="Evaluating")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device)
        pred_dict = model(input_dict)
        # evaluate on real target images:
        # b, t, v, c, h, w
        b, t, v, height, width = target_dict["target_depth"].shape
        gt_depth = target_dict["target_depth"].view(b * t, -1, height, width)
        num_imgs = gt_depth.shape[0]
        valid_depth_mask = gt_depth > 0.0
        rendered_results = pred_dict["render_results"]
        if args.load_ground:
            gt_ground_mask = target_dict["target_ground_masks"].view(b * t, -1, height, width)
            gt_ground_mask = gt_ground_mask.bool()
        num_valid_samples = 0
        eval_flow = (
            "rendered_flow" in rendered_results
            and args.decoder_type == "dummy"
            and args.load_flow
            and "target_flow" in target_dict
        )
        if eval_flow:
            gt_flow = target_dict["target_flow"].view(b * t, -1, height, width, 3)
            pred_flow = rendered_results["rendered_flow"].view(b * t, -1, height, width, 3)
            # pred_flow = gt_flow.clone() + torch.rand_like(gt_flow) * 0.05
            flow_epes, flow_accs_strict, flow_accs_relax, flow_angles = [], [], [], []
            flow_rmse = []

            for i in range(num_imgs):
                if torch.max(gt_flow.norm(dim=-1)) > 1.0:
                    if args.load_ground:
                        non_ground_gt_flow = gt_flow[i][~gt_ground_mask[i] & valid_depth_mask[i]]
                        non_ground_pred_flow = pred_flow[i][
                            ~gt_ground_mask[i] & valid_depth_mask[i]
                        ]
                    else:
                        non_ground_gt_flow = gt_flow[i][valid_depth_mask[i]]
                        non_ground_pred_flow = pred_flow[i][valid_depth_mask[i]]
                    flow_metrics = compute_scene_flow_metrics(
                        non_ground_pred_flow, non_ground_gt_flow
                    )
                    flow_epes.append(flow_metrics["EPE3D"])
                    flow_accs_strict.append(flow_metrics["acc3d_strict"] * 100)
                    flow_accs_relax.append(flow_metrics["acc3d_relax"] * 100)
                    flow_angles.append(flow_metrics["angle_error"])
                    flow_rmse.append(
                        torch.sqrt(
                            F.mse_loss(
                                pred_flow[i][valid_depth_mask[i]],
                                gt_flow[i][valid_depth_mask[i]],
                            )
                        ).item()
                    )
                    num_valid_samples += 1

            flow_epe_sum = np.sum(flow_epes)
            flow_acc_strict_sum = np.sum(flow_accs_strict)
            flow_acc_relax_sum = np.sum(flow_accs_relax)
            flow_angle_sum = np.sum(flow_angles)
            flow_rmse_sum = np.sum(flow_rmse)
            valid_flow_samples = num_valid_samples

            # Update running sums and counts
            total_flow_epes += flow_epe_sum
            total_flow_accs_strict += flow_acc_strict_sum
            total_flow_accs_relax += flow_acc_relax_sum
            total_flow_angles += flow_angle_sum
            total_flow_rmse += flow_rmse_sum
            total_numb_flow_samples += valid_flow_samples

        pbar.set_postfix(
            avg_flow_epe=total_flow_epes / total_numb_flow_samples,
            avg_flow_acc_relax=total_flow_accs_relax / total_numb_flow_samples,
            avg_flow_acc_strict=total_flow_accs_strict / total_numb_flow_samples,
            avg_flow_angle=total_flow_angles / total_numb_flow_samples,
            avg_flow_rmse=total_flow_rmse / total_numb_flow_samples,
        )

    # Create tensors for sums and counts
    result = None
    if eval_flow:
        total_flow_epes_tensor = torch.tensor(total_flow_epes, device=device)
        total_flow_accs_strict_tensor = torch.tensor(total_flow_accs_strict, device=device)
        total_flow_accs_relax_tensor = torch.tensor(total_flow_accs_relax, device=device)
        total_flow_angles_tensor = torch.tensor(total_flow_angles, device=device)
        total_flow_rmse_tensor = torch.tensor(total_flow_rmse, device=device)
        total_numb_flow_samples_tensor = torch.tensor(total_numb_flow_samples, device=device)

        torch.cuda.synchronize()

        if distributed.is_enabled():
            # Aggregate sums across all processes
            torch.distributed.all_reduce(total_flow_epes_tensor)
            torch.distributed.all_reduce(total_flow_accs_strict_tensor)
            torch.distributed.all_reduce(total_flow_accs_relax_tensor)
            torch.distributed.all_reduce(total_flow_angles_tensor)
            torch.distributed.all_reduce(total_flow_rmse_tensor)
            torch.distributed.all_reduce(total_numb_flow_samples_tensor)
        if distributed.is_main_process() and total_numb_flow_samples_tensor.item() > 0:
            avg_flow_epe = total_flow_epes_tensor.item() / total_numb_flow_samples_tensor.item()
            avg_flow_acc_strict = (
                total_flow_accs_strict_tensor.item() / total_numb_flow_samples_tensor.item()
            )
            avg_flow_acc_relax = (
                total_flow_accs_relax_tensor.item() / total_numb_flow_samples_tensor.item()
            )
            avg_flow_angle = total_flow_angles_tensor.item() / total_numb_flow_samples_tensor.item()
            avg_flow_rmse = total_flow_rmse_tensor.item() / total_numb_flow_samples_tensor.item()
            with open(os.path.join(eval_result_dir, f"eval_{name_str}_flow.txt"), "w") as f:
                f.write(f"Average Flow EPE: {avg_flow_epe:.4f}\n")
                f.write(f"Average Flow Acc Strict: {avg_flow_acc_strict:.4f}\n")
                f.write(f"Average Flow Acc Relax: {avg_flow_acc_relax:.4f}\n")
                f.write(f"Average Flow Angle: {avg_flow_angle:.4f}\n")
                f.write(f"Average Flow RMSE: {avg_flow_rmse:.4f}\n")
            logger.info("Evaluation results saved.")
            logger.info(f"Evaluated on {total_numb_flow_samples_tensor.item()} samples.")
            logger.info(
                f"Average Flow EPE: {avg_flow_epe:.4f}, Average Flow Acc Strict: {avg_flow_acc_strict:.4f}, Average Flow Acc Relax: {avg_flow_acc_relax:.4f}, Average Flow Angle: {avg_flow_angle:.4f}"
            )
            logger.info(f"Average Flow RMSE: {avg_flow_rmse:.4f}")
            result = {
                "flow_epe": avg_flow_epe,
                "flow_acc_strict": avg_flow_acc_strict,
                "flow_acc_relax": avg_flow_acc_relax,
                "flow_angle": avg_flow_angle,
                "flow_rmse": avg_flow_rmse,
            }
    torch.cuda.empty_cache()
    return result
