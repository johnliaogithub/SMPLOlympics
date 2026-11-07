# Paper figure: a filmstrip (frames in sequence) of one bout, plus the matching mp4.
#
# ours: en garde -> lunge -> touch -> recovery. The bout starts in lunge range, holds the stand drill
#       for env.film_guard_steps, then lunges; the landed lunge hands over to stand (recovery).
#       Frames: last guard step, mid-lunge, the touch, and the end of recovery (auto, from events).
# flat: frames at env.film_steps of a free self-play bout.
#
#   bash scripts/fencing/paper/filmstrip.sh ours output/HumanoidIm/paper_drills_s0/Humanoid.pth ours
#   bash scripts/fencing/paper/filmstrip.sh flat output/HumanoidIm/paper_flat_s0/Humanoid.pth flat
# Output: output/paper/figures/filmstrip_<tag>.png/.pdf and output/paper/video/bout_<tag>.mp4
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_task, FlatPolicy  # noqa: E402

import numpy as np
import torch
import imageio
import hydra
from isaacgym import gymapi
from omegaconf import DictConfig
from PIL import Image

from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_STAND, D_LUNGE_UPPER, D_LUNGE_GROIN, DRILL_NAMES


_FIXED_CAM = {}


def grab(task, cam_dist, fixed=True):
    gym, sim, env0, cam = task.gym, task.sim, task.envs[0], task.recorder_camera_handles[0]
    # side-on, perpendicular to the line between the fencers, aimed at their midpoint. fixed=True keeps
    # the first frame's pose for the whole clip (a tracking camera makes filmstrip frames incomparable).
    if not fixed or "mid" not in _FIXED_CAM:
        r0 = task._humanoid_root_states_list[0][0, 0:3].cpu().numpy()
        r1 = task._humanoid_root_states_list[1][0, 0:3].cpu().numpy()
        d = r1[:2] - r0[:2]
        _FIXED_CAM["mid"] = 0.5 * (r0 + r1)
        _FIXED_CAM["n"] = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
    mid, n = _FIXED_CAM["mid"], _FIXED_CAM["n"]
    gym.set_camera_location(cam, env0, gymapi.Vec3(mid[0] + cam_dist * n[0], mid[1] + cam_dist * n[1], 1.2),
                            gymapi.Vec3(mid[0], mid[1], 0.9))
    if task.device != 'cpu':
        gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    img = gym.get_camera_image(sim, env0, cam, gymapi.IMAGE_COLOR)
    return img.reshape(img.shape[0], -1, 4)[..., :3].copy()


def compose(frames, captions, out_png, crop=0.2, top=0.28, bottom=0.12):
    """Frames side by side, cropped at left/right, captions underneath (paper style: white, black text)."""
    from PIL import ImageDraw, ImageFont
    tiles = []
    for f in frames:
        h, w = f.shape[:2]
        c = int(w * crop)
        tiles.append(Image.fromarray(f[int(h * top):h - int(h * bottom), c:w - c]))
    W, H = tiles[0].size
    gap, cap_h = 8, max(36, H // 9)
    font = None
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            font = ImageFont.truetype(p, int(cap_h * 0.55))
    font = font or ImageFont.load_default()
    canvas = Image.new("RGB", (len(tiles) * W + (len(tiles) - 1) * gap, H + cap_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for k, (t, cap) in enumerate(zip(tiles, captions)):
        x = k * (W + gap)
        canvas.paste(t, (x, 0))
        tw = draw.textlength(cap, font=font)
        draw.text((x + (W - tw) / 2, H + cap_h * 0.18), cap, fill=(26, 26, 26), font=font)
    canvas.save(out_png)
    canvas.save(out_png.replace(".png", ".pdf"), resolution=300)
    print(f"[film] {out_png} (+ .pdf)")


@torch.no_grad()
def run_ours(cfg, task, ckpt, writer, cam_dist):
    low = FrozenLowLevelPolicy(ckpt, NUM_DRILLS, device=task.device)
    guard = int(cfg.env.get("film_guard_steps", 40))
    recover = int(cfg.env.get("film_recover_steps", 60))
    lunge = D_LUNGE_GROIN if cfg.env.get("film_target", "upper") == "groin" else D_LUNGE_UPPER
    probs = torch.zeros(NUM_DRILLS, device=task.device)
    probs[lunge] = 1.0                       # spawn in lunge range
    task.drill_probs = probs
    task.reset()
    frames, hit_t, t = [], None, 0
    while True:
        cmd = D_STAND if t < guard else lunge
        if t == guard:
            task._lunge_landed[:] = False
        if cmd == lunge and bool(task._lunge_landed[0]):
            cmd = D_STAND
        task.drill_ids[:] = cmd
        task.opp_drill_ids[:] = D_STAND
        task.opp_frozen[:] = True
        obs = task.obs_buf
        task.step(torch.cat([low(obs[:1], task.drill_ids), low(obs[1:2], task.opp_drill_ids)], 0))
        img = grab(task, cam_dist)
        frames.append(img)
        writer.append_data(img)
        if hit_t is None and t >= guard and bool(task._lunge_landed[0]):
            hit_t = t
        t += 1
        if bool(task.reset_buf[0]) or (hit_t is not None and t >= hit_t + recover) or t >= guard + 70 + recover:
            break
    if hit_t is None:
        print("[film] WARNING: the lunge never landed; using evenly spaced frames")
        idx = np.linspace(guard - 1, len(frames) - 1, 4).astype(int).tolist()
        return [frames[i] for i in idx], ["En garde", "Lunge", "Lunge", "End"], idx
    mid = guard + max(1, (hit_t - guard) // 2)
    idx = [guard - 1, mid, hit_t, len(frames) - 1]
    return [frames[i] for i in idx], ["En garde", "Lunge", "Touch", "Recovery"], idx


@torch.no_grad()
def run_flat(cfg, task, ckpt, writer, cam_dist):
    pol0 = FlatPolicy(ckpt, device=task.device)
    op = ckpt.replace(".pth", "_op.pth")
    pol1 = FlatPolicy(op, device=task.device) if os.path.exists(op) else pol0
    steps = [int(s) for s in cfg.env.get("film_steps", [20, 60, 100, 140])]
    # the baseline env picks its curriculum stage from step_counter; stage 0 disables win endings, so
    # start in the final stage, as at the end of training
    task.step_counter = max(task.warmup_time + 10, task.curriculum_stages[-1]["step_threshold"] + 1)
    task.reset()
    frames = []
    for t in range(max(steps) + 1):
        obs = task.obs_buf
        task.step(torch.cat([pol0(obs[:1]), pol1(obs[1:2])], 0))
        img = grab(task, cam_dist)
        frames.append(img)
        writer.append_data(img)
        if bool(task.reset_buf[0]):
            print(f"[film] bout ended at step {t}")
            break
    if len(frames) - 1 < max(steps):
        # the bout ended early (e.g. a short win rule): spread the frames over the whole bout instead
        steps = np.linspace(0, len(frames) - 1, len(steps)).round().astype(int).tolist()
    return [frames[s] for s in steps], [f"t = {s / 30:.1f} s" for s in steps], steps


@hydra.main(version_base=None, config_path="../data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg, task = make_task(cfg_hydra, test=True, seed=int(cfg_hydra.get("seed", 0)))
    if task.recorder_camera_handles[0] == -1:
        raise RuntimeError("Recorder camera missing: launch with +record_video=True")
    tag, method = cfg.env.film_tag, cfg.env.film_method
    os.makedirs("output/paper/figures", exist_ok=True)
    os.makedirs("output/paper/video", exist_ok=True)
    writer = imageio.get_writer(f"output/paper/video/bout_{tag}.mp4", fps=30, macro_block_size=None)
    cam_dist = float(cfg.env.get("film_cam_dist", 4.5))
    frames, caps, idx = (run_ours if method == "ours" else run_flat)(cfg, task, cfg.env.film_checkpoint, writer, cam_dist)
    writer.close()
    print(f"[film] video: output/paper/video/bout_{tag}.mp4 | frames at steps {idx}")
    compose(frames, caps, f"output/paper/figures/filmstrip_{tag}.png")
    for k, f in zip(idx, frames):
        Image.fromarray(f).save(f"output/paper/figures/filmstrip_{tag}_t{k:03d}.png")


if __name__ == "__main__":
    main()
