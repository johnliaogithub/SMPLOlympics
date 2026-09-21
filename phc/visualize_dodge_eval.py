# Diagnostic: does the dueling-dodge policy generalize, or is it one fixed exchange?
# Runs N dodge bouts (learner = dodge/duel, opponent = lunge) with randomized spawns (the SAME
# dodge_spawn_jitter as training), the first half vs an UPPER lunge and the second vs GROIN, and
# records them back-to-back (no end pause). Prints the learner's net lateral shift (dx) per bout,
# so you can see at a glance whether every dodge goes the same direction.
#
#   bash scripts/fencing/record_dodge_eval.sh [low_level_checkpoint]

import os, sys
os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')

from isaacgym import gymapi  # noqa: F401 (before torch)
import torch, imageio
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.visualize_drills import label_frame
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_DODGE, D_LUNGE_UPPER, D_LUNGE_GROIN


@torch.no_grad()
def render(cfg, task, low_level, out_path):
    gym, sim = task.gym, task.sim
    cam = task.recorder_camera_handles[0]
    env0 = task.envs[0]
    if cam == -1:
        raise RuntimeError("Recorder camera not created — launch with +record_video=True")
    N = task.num_envs
    n_bouts = int(cfg.env.get("n_eval_bouts", 10))
    max_steps = int(cfg.env.get("strike_episode_length", 70)) + 20

    writer = imageio.get_writer(out_path, fps=30, macro_block_size=None)
    summary = []
    for bout in range(n_bouts):
        target = D_LUNGE_UPPER if bout < n_bouts // 2 else D_LUNGE_GROIN
        tname = "UPPER" if target == D_LUNGE_UPPER else "GROIN"
        probs = torch.zeros(NUM_DRILLS, device=task.device); probs[D_DODGE] = 1.0
        task.drill_probs = probs
        task.reset()   # applies dodge_spawn_jitter, samples a fresh spawn gap
        x0 = float(task._humanoid_root_states_list[0][0, 0].item())
        y0 = float(task._humanoid_root_states_list[0][0, 1].item())
        dx = dy = 0.0
        for t in range(max_steps):
            task.drill_ids[:] = D_DODGE
            task.opp_drill_ids[:] = target
            task.opp_frozen[:] = False
            obs = task.obs_buf
            z0 = low_level(obs[:N], task.drill_ids)
            z1 = low_level(obs[N:2 * N], task.opp_drill_ids)
            task.step(torch.cat([z0, z1], dim=0))
            lx = float(task._humanoid_root_states_list[0][0, 0].item())
            ly = float(task._humanoid_root_states_list[0][0, 1].item())
            dx, dy = lx - x0, ly - y0
            r0 = task._humanoid_root_states_list[0][0, 0:3].cpu().numpy()
            r1 = task._humanoid_root_states_list[1][0, 0:3].cpu().numpy()
            mid = (r0 + r1) / 2.0
            gym.set_camera_location(cam, env0,
                                    gymapi.Vec3(mid[0] + 5.0, mid[1], 2.0),
                                    gymapi.Vec3(mid[0], mid[1], 1.0))
            if task.device != 'cpu':
                gym.fetch_results(sim, True)
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env0, cam, gymapi.IMAGE_COLOR)
            img = img.reshape(img.shape[0], -1, 4)[..., :3]
            label = f"bout {bout+1}/{n_bouts}  opp lunge {tname}   dodge dx={dx:+.2f} dy={dy:+.2f}"
            writer.append_data(label_frame(img, "dodge eval", label))
            if bool(task.reset_buf[0].item()):
                break
        side = "LEFT" if dx < -0.05 else ("RIGHT" if dx > 0.05 else "center")
        summary.append((bout + 1, tname, round(dx, 2), round(dy, 2), side))
        print(f"[bout {bout+1}] opp={tname}  dx={dx:+.2f}  dy={dy:+.2f}  -> {side}")
    writer.close()
    print("\n=== dodge-direction summary (dx = lateral, dy = along the strip) ===")
    for b, tn, ddx, ddy, side in summary:
        print(f"  bout {b:2d}  opp {tn:5s}  dx={ddx:+.2f}  dy={ddy:+.2f}  {side}")
    lefts = sum(1 for *_, s in summary if s == "LEFT")
    rights = sum(1 for *_, s in summary if s == "RIGHT")
    print(f"  LEFT={lefts}  RIGHT={rights}  center={len(summary)-lefts-rights}")
    print(f"\n============ dodge eval saved to {out_path} ============")


@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()
    flags.test = True
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
    args = EasyDict({"task": cfg.env.task, "device_id": cfg.device_id, "rl_device": cfg.rl_device,
                     "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
                     "headless": cfg.headless, "device": cfg.device})
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    low_level = FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device)
    os.makedirs("output/renderings", exist_ok=True)
    render(cfg, task, low_level, os.path.join("output/renderings", "dodge_eval.mp4"))


if __name__ == '__main__':
    main()
