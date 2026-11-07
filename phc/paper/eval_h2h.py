# Paper evaluation: head-to-head bouts, strategy (+ frozen drills policy) vs the flat self-play baseline.
#
# Env: HumanoidFencingStrategyZ (same touch / out-of-bounds / fall rules as strategy training). Both
# fencers spawn 3 m apart (drill_probs = stand only, the flat baseline's default spawn gap).
# The strategy picks a drill every macro_K steps (sampled, as in training) and re-picks at every new
# bout; the frozen drills policy turns (obs, drill) into PULSE latents. The flat policy reads the first
# 512 obs dims (the shared fencing observation; the drills env appends only its own drill one-hot).
# Sides: env.h2h_strategy_side = 0 or 1 (which agent the strategy controls); run both.
# env.h2h_strategy_side = 2: strategy self-play (both fencers use strategy + drills), for the video only.
# Body contact (root gap < contact_term_dist) ends a bout with no winner; contact_term_dist=0 disables it.
#
#   bash scripts/fencing/paper/eval_h2h.sh <strategy seed> <side> <tag> [overrides]
# Appends (or replaces) a row in output/paper/h2h.csv.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_task, FlatPolicy  # noqa: E402

import torch  # noqa: E402
import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from phc.train_fencing_strategy import FrozenLowLevelPolicy, StrategyNet  # noqa: E402
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, DRILL_NAMES  # noqa: E402
from eval_motion import write_row  # noqa: E402


@hydra.main(version_base=None, config_path="../data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg, task = make_task(cfg_hydra, test=True, seed=int(cfg_hydra.get("seed", 0)))
    e = cfg.env
    dev, N, K = task.device, task.num_envs, int(e.get("macro_K", 15))
    side = int(e.h2h_strategy_side)
    low = FrozenLowLevelPolicy(e.low_level_checkpoint, NUM_DRILLS, device=dev)
    ck = torch.load(e.h2h_strategy_checkpoint, map_location=dev)
    net = StrategyNet(task.get_obs_size(), NUM_DRILLS).to(dev)
    net.load_state_dict(ck["strategy_net"])
    net.eval()
    flat = FlatPolicy(e.h2h_flat_checkpoint, device=dev)
    flat_dim = flat.mean.shape[0]
    print(f"[h2h] strategy {e.h2h_strategy_checkpoint} (iter {ck.get('iter')}) as agent {side} | flat obs_dim {flat_dim}")

    task.win_frame_hit = float(e.get("win_frame_hit", 2))     # end value of the training ramp
    task.step_counter = task.warmup_time + 10
    task.reset()
    s_rows, f_rows = (slice(0, N), slice(N, 2 * N)) if side != 1 else (slice(N, 2 * N), slice(0, N))
    drills = torch.zeros(N, dtype=torch.long, device=dev)
    drills_b = torch.zeros(N, dtype=torch.long, device=dev)   # agent 1's drills in self-play (side 2)
    drill_log = []
    k = torch.zeros(N, dtype=torch.long, device=dev)          # steps since the last strategy decision
    names = ["strategy_touch", "flat_touch", "double_touch", "contact", "strategy_oob", "flat_oob",
             "strategy_fall", "flat_fall", "timeout"]
    tally = {n: 0 for n in names}
    usage = torch.zeros(NUM_DRILLS, device=dev)
    blen_sum, bouts, T = 0, 0, int(e.get("h2h_steps", 1500))
    blen = torch.zeros(N, device=dev)
    head = int(task._head_id[0])
    # optional video of env 0 (launch with env.num_envs=1 +record_video=True +env.h2h_video=<tag>)
    vid = e.get("h2h_video", None)
    if vid:
        import imageio
        from filmstrip import grab, compose
        os.makedirs("output/paper/video", exist_ok=True)
        writer = imageio.get_writer(f"output/paper/video/{vid}.mp4", fps=30, macro_block_size=None)
        frames, first_end = [], None
    with torch.no_grad():
        for t in range(T):
            obs = torch.clamp(task.obs_buf, -5.0, 5.0)
            redo = k == 0
            if redo.any():
                logits, _ = net(obs[s_rows])
                pick = torch.distributions.Categorical(logits=logits).sample()
                drills[redo] = pick[redo]
                usage += torch.bincount(pick[redo], minlength=NUM_DRILLS).float()
                if side == 2:
                    logits_b, _ = net(obs[f_rows])
                    pick_b = torch.distributions.Categorical(logits=logits_b).sample()
                    drills_b[redo] = pick_b[redo]
            if side == 2:
                task.drill_ids[:] = drills
                task.opp_drill_ids[:] = drills_b
                task.step(torch.cat([low(obs[s_rows], drills), low(obs[f_rows], drills_b)], 0))
            else:
                if side == 0:
                    task.drill_ids[:] = drills
                else:
                    task.opp_drill_ids[:] = drills
                z_s = low(obs[s_rows], drills)
                z_f = flat(obs[f_rows][:, :flat_dim])
                task.step(torch.cat([z_s, z_f] if side == 0 else [z_f, z_s], 0))
            drill_log.append((DRILL_NAMES[int(drills[0])], DRILL_NAMES[int(drills_b[0])] if side == 2 else ""))
            if vid:
                img = grab(task, float(e.get("film_cam_dist", 4.5)))
                writer.append_data(img)
                frames.append(img)
            k = (k + 1) % K
            blen += 1
            done = task.reset_buf.bool()
            if t < 2:          # first steps after the initial reset are not real bouts
                done[:] = False
            if vid and first_end is None and t >= 2 and bool(task.reset_buf[0]):
                first_end = t
            if done.any():
                g, r = task.green_win.bool(), task.red_win.bool()
                s_win, f_win = (g, r) if side == 0 else (r, g)
                hz = [task._rigid_body_pos_list[i][:, head, 2] < task.fall_head_height for i in range(2)]
                rs = [task._humanoid_root_states_list[i] for i in range(2)]
                bb = task.bounding_box
                oob = [(x[:, 0] < bb[0]) | (x[:, 0] > bb[1]) | (x[:, 1] < bb[2]) | (x[:, 1] > bb[3]) for x in rs]
                s_i, f_i = (0, 1) if side == 0 else (1, 0)
                touch = s_win | f_win
                rest = done & ~touch
                c = {
                    "strategy_touch": done & s_win & ~f_win, "flat_touch": done & f_win & ~s_win,
                    "double_touch": done & s_win & f_win,
                    "contact": rest & task._body_contact,
                    "strategy_oob": rest & ~task._body_contact & oob[s_i],
                    "flat_oob": rest & ~task._body_contact & ~oob[s_i] & oob[f_i],
                }
                left = rest & ~task._body_contact & ~oob[0] & ~oob[1]
                c["strategy_fall"] = left & hz[s_i]
                c["flat_fall"] = left & ~hz[s_i] & hz[f_i]
                c["timeout"] = left & ~hz[0] & ~hz[1]
                for n in names:
                    tally[n] += int(c[n].sum())
                blen_sum += float(blen[done].sum())
                bouts += int(done.sum())
                blen[done] = 0
                k[done] = 0
                ids = done.nonzero(as_tuple=False).flatten()
                task.reset(ids)
    if vid:
        writer.close()
        end = first_end if first_end is not None else len(frames) - 1
        idx = [int(round(x)) for x in torch.linspace(0, end, 4).tolist()]
        os.makedirs("output/paper/figures", exist_ok=True)
        def cap(i):
            a, b = drill_log[i]
            pretty = lambda x: x.replace("_", " ").replace("stand", "en garde").replace("lunge upper", "lunge").replace("lunge groin", "lunge low")
            return f"t = {i / 30:.1f} s: " + (f"{pretty(a)} / {pretty(b)}" if b else pretty(a))
        compose([frames[i] for i in idx], [cap(i) for i in idx], f"output/paper/figures/filmstrip_{vid}.png")
        print(f"[h2h] video output/paper/video/{vid}.mp4, first bout ends at step {first_end}")
        return
    b = max(bouts, 1)
    row = {"tag": e.h2h_tag, "strategy_ckpt": e.h2h_strategy_checkpoint, "strategy_side": side,
           "contact_term_dist": task.contact_term_dist, "bouts": bouts, "mean_bout_steps": blen_sum / b}
    row.update({n: tally[n] / b for n in names})
    decided = tally["strategy_touch"] + tally["flat_touch"]
    row["strategy_share_of_touches"] = tally["strategy_touch"] / max(decided, 1)
    u = usage / usage.sum().clamp_min(1)
    row.update({f"use_{DRILL_NAMES[i]}": float(u[i]) for i in range(NUM_DRILLS)})
    for kk, v in row.items():
        print(f"  {kk:26s} {v}")
    write_row(row, "output/paper/h2h.csv")


if __name__ == "__main__":
    main()
