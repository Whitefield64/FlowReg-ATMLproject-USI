"""Side-by-side demo video: Baseline PPO vs FlowReg PPO on MiniGrid-FourRooms."""

from __future__ import annotations

import argparse
from pathlib import Path

import minigrid  # noqa: F401 — registers MiniGrid environments with gymnasium
import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def make_render_env(env_id: str, seed: int) -> gym.Env:
    """Create a MiniGrid env that returns RGB frames from render(), wrapped for PPO inference."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_side_by_side_frame(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    label_a: str,
    label_b: str,
    ep: int,
    step: int,
    rew_a: float,
    rew_b: float,
    done_a: bool,
    done_b: bool,
    target_h: int = 256,
) -> np.ndarray:
    """Concatenate two game frames side-by-side with banner and reward info."""
    img_a = Image.fromarray(frame_a)
    img_b = Image.fromarray(frame_b)

    w_a = int(img_a.width * target_h / img_a.height)
    w_b = int(img_b.width * target_h / img_b.height)
    img_a = img_a.resize((w_a, target_h), Image.NEAREST)
    img_b = img_b.resize((w_b, target_h), Image.NEAREST)

    banner_h = 44
    info_h = 34
    sep_w = 4
    total_w = w_a + sep_w + w_b
    total_h = banner_h + target_h + info_h

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # Separator bar
    draw.rectangle([w_a, 0, w_a + sep_w - 1, total_h], fill=(60, 60, 60))

    # Agent name banners
    draw.text((w_a // 2, banner_h // 2), label_a, fill=(255, 200, 80), anchor="mm")
    draw.text((w_a + sep_w + w_b // 2, banner_h // 2), label_b, fill=(80, 200, 255), anchor="mm")

    # Game frames
    canvas.paste(img_a, (0, banner_h))
    canvas.paste(img_b, (w_a + sep_w, banner_h))

    # Info bar: step and cumulative reward per agent
    info_y = banner_h + target_h
    text_a = f"DONE  R = {rew_a:.2f}" if done_a else f"Ep {ep + 1}  step {step}  R = {rew_a:.2f}"
    text_b = f"DONE  R = {rew_b:.2f}" if done_b else f"Ep {ep + 1}  step {step}  R = {rew_b:.2f}"
    draw.text((w_a // 2, info_y + info_h // 2), text_a, fill=(200, 200, 200), anchor="mm")
    draw.text((w_a + sep_w + w_b // 2, info_y + info_h // 2), text_b, fill=(200, 200, 200), anchor="mm")

    return np.array(canvas)


def run_episodes(
    model_a: PPO,
    model_b: PPO,
    env_a: gym.Env,
    env_b: gym.Env,
    n_episodes: int,
    max_steps: int,
    label_a: str,
    label_b: str,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Run both agents step-by-step, collecting side-by-side frames for every step."""
    all_frames: list[np.ndarray] = []
    episode_rewards: list[tuple[float, float]] = []

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()

    for ep in range(n_episodes):
        done_a = done_b = False
        rew_a = rew_b = 0.0
        step = 0
        last_frame_a = env_a.render()
        last_frame_b = env_b.render()

        while step < max_steps and not (done_a and done_b):
            if not done_a:
                action_a, _ = model_a.predict(obs_a, deterministic=True)
                obs_a, r_a, term_a, trunc_a, _ = env_a.step(int(action_a))
                rew_a += float(r_a)
                last_frame_a = env_a.render()
                done_a = bool(term_a) or bool(trunc_a)

            if not done_b:
                action_b, _ = model_b.predict(obs_b, deterministic=True)
                obs_b, r_b, term_b, trunc_b, _ = env_b.step(int(action_b))
                rew_b += float(r_b)
                last_frame_b = env_b.render()
                done_b = bool(term_b) or bool(trunc_b)

            step += 1
            all_frames.append(
                make_side_by_side_frame(
                    last_frame_a, last_frame_b, label_a, label_b,
                    ep, step, rew_a, rew_b, done_a, done_b,
                )
            )

        episode_rewards.append((rew_a, rew_b))
        print(f"  Ep {ep + 1}: {label_a} R={rew_a:.2f}  |  {label_b} R={rew_b:.2f}  ({step} steps)")

        if ep < n_episodes - 1:
            obs_a, _ = env_a.reset()
            obs_b, _ = env_b.reset()

    return all_frames, episode_rewards


def save_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    import imageio

    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side MP4 demo: baseline PPO vs FlowReg PPO."
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline PPO .zip checkpoint")
    parser.add_argument("--flowreg", required=True, help="Path to FlowReg PPO .zip checkpoint")
    parser.add_argument("--env-id", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="demo_fourrooms.mp4")
    parser.add_argument("--label-baseline", default="Baseline PPO")
    parser.add_argument("--label-flowreg", default="FlowReg PPO")
    args = parser.parse_args()

    print(f"Loading baseline:  {args.baseline}")
    model_baseline = PPO.load(args.baseline, device="cpu")
    print(f"Loading flowreg:   {args.flowreg}")
    model_flowreg = PPO.load(args.flowreg, device="cpu")

    print(f"Creating environments ({args.env_id}, seed={args.seed}) ...")
    env_a = make_render_env(args.env_id, seed=args.seed)
    env_b = make_render_env(args.env_id, seed=args.seed)

    print(f"Running {args.episodes} episodes (max {args.max_steps} steps each) ...")
    frames, rewards = run_episodes(
        model_baseline, model_flowreg,
        env_a, env_b,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        label_a=args.label_baseline,
        label_b=args.label_flowreg,
    )

    env_a.close()
    env_b.close()

    output = Path(args.output)
    print(f"\nSaving {len(frames)} frames → {output} at {args.fps} fps ...")
    save_mp4(frames, output, fps=args.fps)

    print("\nEpisode summary:")
    for i, (r_a, r_b) in enumerate(rewards):
        winner = args.label_baseline if r_a >= r_b else args.label_flowreg
        print(f"  Ep {i + 1}: {args.label_baseline} {r_a:.2f}  |  {args.label_flowreg} {r_b:.2f}  → {winner} wins")

    print(f"\nDone! Video saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
