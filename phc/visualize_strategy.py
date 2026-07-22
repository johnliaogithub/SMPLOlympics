# Record a strategy-net fencing bout: the high-level policy picks a drill every
# macro_K steps for BOTH fencers, the frozen low-level executes it, and each frame is
# labeled with the drill each agent is currently doing so you can see the decisions.
#
# Example:
#   bash scripts/fencing/record_strategy.sh

import os
import sys

os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd())
sys.path.append('./SMPLSim')

from isaacgym import gymapi  # noqa: F401  (before torch)
import torch
import imageio
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy, StrategyNet
from phc.visualize_drills import label_frame
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, DRILL_NAMES


@torch.no_grad()
def render_bout(cfg, task, low_level, net, out_path):
    gym, sim = task.gym, task.sim
    cam = task.recorder_camera_handles[0]
    env0 = task.envs[0]
    if cam == -1:
        raise RuntimeError("Recorder camera not created — launch with +record_video=True")
    N = task.num_envs
    macro_K = task.macro_K
    total_steps = int(cfg.env.get("clip_len", 600))   # ~20 s @ 30 Hz

    writer = imageio.get_writer(out_path, fps=30, macro_block_size=None)
    task.reset()
    d0 = d1 = 0
    prev_img = None   # last live frame, held as the freeze when the bout ends

    for t in range(total_steps):
        obs = task.obs_buf
        if t % macro_K == 0:
            # both fencers use the SAME strategy net (deterministic: argmax)
            d0 = net(torch.clamp(obs[:N], -5.0, 5.0))[0].argmax(-1)
            d1 = net(torch.clamp(obs[N:2 * N], -5.0, 5.0))[0].argmax(-1)
            task.drill_ids[:] = d0
            task.opp_drill_ids[:] = d1
            task.opp_frozen[:] = False   # both fencers active in a real bout

        z0 = low_level(obs[:N], task.drill_ids)
        z1 = low_level(obs[N:2 * N], task.opp_drill_ids)
        task.step(torch.cat([z0, z1], dim=0))

        # End the clip at the FIRST bout termination. The env has already auto-reset the
        # fencers to spawn this step, so rendering now would show a teleport — instead
        # freeze the previous (live) frame, labeled with why the bout ended.
        if bool(task.reset_buf[0].item()):
            if bool(task.green_win[0].item()):
                verdict = "GREEN (agent 0) SCORES"
            elif bool(task.red_win[0].item()):
                verdict = "RED (opponent) SCORES"
            elif bool(task._body_contact[0].item()):
                verdict = "BODY CONTACT -- both penalized"
            elif bool(task.out_bound[0].item()):
                verdict = "OUT OF BOUNDS"
            else:
                verdict = "TIMEOUT"
            print(f"[bout] ended at step {t}: {verdict}")
            if prev_img is not None:
                for _ in range(30):   # ~1 s freeze on the ending
                    writer.append_data(label_frame(prev_img, "BOUT OVER", verdict))
            break

        # camera: side-on, centered on the midpoint of the two fencers
        r0 = task._humanoid_root_states_list[0][0, 0:3].cpu().numpy()
        r1 = task._humanoid_root_states_list[1][0, 0:3].cpu().numpy()
        mid = (r0 + r1) / 2.0
        gym.set_camera_location(cam, env0,
                                gymapi.Vec3(mid[0] + 5.0, mid[1], 2.0),
                                gymapi.Vec3(mid[0], mid[1], 1.0))
        if task.device != 'cpu':
            gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, env0, cam, gymapi.IMAGE_COLOR)
        img = img.reshape(img.shape[0], -1, 4)[..., :3]

        label = f"green: {DRILL_NAMES[int(d0[0])]}   |   red: {DRILL_NAMES[int(d1[0])]}"
        writer.append_data(label_frame(img, "strategy bout", label))
        prev_img = img

    writer.close()
    print(f"\n============ Strategy bout saved to {out_path} ============")


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
    args = EasyDict({
        "task": cfg.env.task, "device_id": cfg.device_id, "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless, "device": cfg.device,
    })
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    task.low_level_policy = FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device)
    net = StrategyNet(task.get_obs_size(), NUM_DRILLS).to(task.device)
    ck = torch.load(cfg.env["strategy_checkpoint"], map_location=task.device)
    net.load_state_dict(ck["strategy_net"])
    net.eval()
    print(f"[Strategy] loaded {cfg.env['strategy_checkpoint']} (iter {ck.get('iter', '?')})")

    os.makedirs("output/renderings", exist_ok=True)
    out_path = os.path.join("output/renderings", f"{cfg.exp_name}_bout.mp4")
    render_bout(cfg, task, task.low_level_policy, net, out_path)


if __name__ == '__main__':
    main()
