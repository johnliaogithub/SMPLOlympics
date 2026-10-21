# Render all six fencing drills into a single labeled video.
#
# Drives the env directly with the FROZEN low-level drill policy, forcing each
# drill in turn, captures the recorder camera, overlays the drill name with PIL,
# and writes one combined mp4 to output/renderings/.
#
# Example:
#   bash scripts/fencing/visualize_drills.sh
#   bash scripts/fencing/visualize_drills.sh output/HumanoidIm/fencing_drills_v1/Humanoid.pth

import os
import sys

os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd())
sys.path.append('./SMPLSim')

from isaacgym import gymapi  # noqa: F401  (before torch)
import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import (
    NUM_DRILLS, DRILL_NAMES, D_STAND, D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE,
)


def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label_frame(rgb, title, subtitle):
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    W = img.width
    f_big, f_small = _font(max(28, W // 22)), _font(max(16, W // 40))
    # translucent banner
    draw.rectangle([0, 0, W, int(img.height * 0.16)], fill=(0, 0, 0))
    draw.text((12, 6), title, fill=(255, 255, 255), font=f_big)
    draw.text((12, 6 + f_big.size + 4), subtitle, fill=(180, 220, 255), font=f_small)
    return np.asarray(img)


@torch.no_grad()
def render_drills(cfg, task, low_level, out_path):
    gym, sim = task.gym, task.sim
    cam = task.recorder_camera_handles[0]
    env0 = task.envs[0]
    if cam == -1:
        raise RuntimeError(
            "Recorder camera was not created (handle -1). Headless runs disable "
            "graphics unless record_video is set — launch with +record_video=True.")
    N = task.num_envs
    clip_len = cfg.env.get("clip_len", 120)   # steps per drill (~4s @ 30Hz)
    # Which drills to render; default = all. e.g. +env.viz_drills=[3,4] for lunges only.
    drills_to_show = cfg.env.get("viz_drills", list(range(NUM_DRILLS)))

    writer = imageio.get_writer(out_path, fps=30, macro_block_size=None)

    for d in drills_to_show:
        is_dodge = (d == D_DODGE)
        opp = D_LUNGE_UPPER if is_dodge else D_STAND
        subtitle = "vs lunging opponent" if is_dodge else "vs idle opponent"
        # Make the reset sample THIS drill so spawn distance + opponent are set
        # correctly by _resample_drills (strike drills spawn in range).
        probs = torch.zeros(NUM_DRILLS, device=task.device)
        probs[d] = 1.0
        task.drill_probs = probs
        task.reset()

        prev_img = None
        for t in range(clip_len):
            # Force the drill every step.
            # (except after a landed lunge switched the bout to stand: lunge_recover_as_stand)
            if not (getattr(task, "lunge_recover_as_stand", False) and bool(task._lunge_landed[0])):
                task.drill_ids[:] = d
            task.opp_drill_ids[:] = opp
            task.opp_frozen[:] = (not is_dodge)

            obs = task.obs_buf
            z0 = low_level(obs[:N], task.drill_ids)
            z1 = low_level(obs[N:2 * N], task.opp_drill_ids)
            task.step(torch.cat([z0, z1], dim=0))

            # Optional trace (+env.viz_trace=True): what the stand clip actually does, step by step.
            if cfg.env.get("viz_trace", False) and d in (D_STAND, D_LUNGE_UPPER, D_LUNGE_GROIN) and (t % 5 == 0 or t == clip_len - 1):
                head = int(task._build_key_body_ids_tensor(["Head"])[0])
                bp0, bp1 = task._rigid_body_pos_list[0][0], task._rigid_body_pos_list[1][0]
                r0, r1 = task._humanoid_root_states_list[0][0, :2], task._humanoid_root_states_list[1][0, :2]
                fwd = (r1 - r0) / (r1 - r0).norm()
                up1 = bp1[task._chest_id[0]] - bp1[task._pelvis_id[0]]; up1 = up1 / up1.norm()
                lean1 = float(((bp1[task._chest_id[0], :2] - bp1[task._pelvis_id[0], :2]) * (-fwd)).sum())
                split0 = float(((bp0[task._right_foot_id[0], :2] - bp0[task._left_foot_id[0], :2]) * fwd).sum())
                up0 = bp0[task._chest_id[0]] - bp0[task._pelvis_id[0]]; up0 = up0 / up0.norm()
                hit0 = bool(task.sword_hit_list[0][0]) if hasattr(task, "sword_hit_list") else False
                print(f"[trace {DRILL_NAMES[d]} t={t:3d}] learner: head_z={float(bp0[head, 2]):.2f} spine_up={float(up0[2]):.2f} "
                      f"hit={hit0} landed={bool(task._lunge_landed[0])} | "
                      f"opp: frozen={bool(task.opp_frozen[0])} head_z={float(bp1[head, 2]):.2f} "
                      f"spine_up={float(up1[2]):.2f} lean_toward_learner={lean1:+.2f}m | learner: split(R ahead of L)={split0:+.2f}m "
                      f"gap={float((r1 - r0).norm()):.2f}m episode_progress={int(task.progress_buf[0])} reset_buf={bool(task.reset_buf[0])}")

            # Episode ended (touch / timeout / fall): stop the live footage and HOLD the last live
            # frame for the remaining frames, so every drill clip is the same total length
            # (clip_len = drill + freeze) instead of shivering in the post-terminal state.
            if bool(task.reset_buf[0].item()) and prev_img is not None:
                for _ in range(clip_len - t):
                    writer.append_data(label_frame(prev_img, f"{d}: {DRILL_NAMES[d]}", subtitle + "   [episode ended]"))
                break

            # Side-on follow camera on agent 0.
            root = task._humanoid_root_states_list[0][0, 0:3].cpu().numpy()
            gym.set_camera_location(
                cam, env0,
                gymapi.Vec3(root[0] + 4.0, root[1], 1.6),
                gymapi.Vec3(root[0], root[1], 0.9),
            )
            if task.device != 'cpu':
                gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env0, cam, gymapi.IMAGE_COLOR)
            img = img.reshape(img.shape[0], -1, 4)[..., :3]

            writer.append_data(label_frame(img, f"{d}: {DRILL_NAMES[d]}", subtitle))
            prev_img = img

        print(f"[viz] rendered drill {d} ({DRILL_NAMES[d]})")

    writer.close()
    print(f"\n============ Combined drill video saved to {out_path} ============")


@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()

    flags.test = True   # enables the render path
    flags.debug = flags.follow = flags.server_mode = False
    flags.im_eval = flags.no_virtual_display = flags.render_o3d = False
    flags.fixed = flags.divide_group = flags.no_collision_check = flags.fixed_path = False
    flags.real_path = flags.show_traj = flags.slow = flags.real_traj = flags.trigger_input = False
    flags.add_proj = cfg.get("add_proj", False)
    flags.has_eval = cfg.get("has_eval", False)

    set_seed(cfg.get("seed", 0), False)

    cfg_train = cfg.learning
    cfg_train['params']['config']["num_actors"] = cfg.env.num_envs

    sim_params = parse_sim_params(cfg)
    args = EasyDict({
        "task": cfg.env.task,
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless,
        "device": cfg.device,
    })
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    low_level = FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device)

    os.makedirs("output/renderings", exist_ok=True)
    out_path = os.path.join("output/renderings", f"{cfg.exp_name}_drills_combined.mp4")
    render_drills(cfg, task, low_level, out_path)


if __name__ == '__main__':
    main()
