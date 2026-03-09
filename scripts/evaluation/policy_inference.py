"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, gr00tn1.6, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument("--policy_device", type=str, default=None, help="Device for policy inference (e.g. cuda:1). Defaults to --device.")
parser.add_argument("--policy_camera_remap", type=str, default=None, help="Remap env camera names for policy, e.g. 'front=camera1,wrist=camera2'")
parser.add_argument("--save_video", action="store_true", help="Save episode videos from the front camera.")
parser.add_argument("--video_dir", type=str, default="logs/videos", help="Directory to save episode videos.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import carb
import gymnasium as gym
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)

import leisaac  # noqa: F401


class EpisodeStepLogger:
    """Logs per-step telemetry to a jsonl file for post-run analysis."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._fh = None
        self._step = 0

    def begin_episode(self, episode_id: int):
        import json  # noqa: F811
        if self._fh:
            self._fh.close()
        path = os.path.join(self.log_dir, f"episode_{episode_id}_steps.jsonl")
        self._fh = open(path, "w")
        self._step = 0
        self._json = json

    def log_step(self, obs_dict: dict, action: torch.Tensor):
        if not self._fh:
            return
        row = {"step": self._step, "t": time.time()}
        for key in ("joint_pos", "joint_vel", "joint_pos_target", "ee_frame_state", "user_vel_state"):
            if key in obs_dict:
                val = obs_dict[key]
                if isinstance(val, torch.Tensor):
                    val = val[0].cpu().tolist()
                row[key] = val
        if isinstance(action, torch.Tensor):
            row["action"] = action.flatten().cpu().tolist()
        self._fh.write(self._json.dumps(row) + "\n")
        self._step += 1

    def end_episode(self):
        if self._fh:
            self._fh.close()
            self._fh = None
        print(f"[StepLog] Saved {self._step} steps -> {self.log_dir}")


class EpisodeVideoRecorder:
    """Records front+wrist side-by-side and a separate viewport (scene) video."""

    def __init__(self, video_dir: str, fps: int = 30):
        os.makedirs(video_dir, exist_ok=True)
        self.video_dir = video_dir
        self.fps = fps
        self._robot_frames: list[np.ndarray] = []
        self._scene_frames: list[np.ndarray] = []
        self._viewport_annotator = None

    def setup_viewport(self):
        """Create a render-product annotator on the viewport camera."""
        try:
            import omni.replicator.core as rep
            rp = rep.create.render_product("/OmniverseKit_Persp", (640, 480))
            self._viewport_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._viewport_annotator.attach([rp])
        except Exception as e:
            print(f"[Video] Viewport capture not available: {e}")
            self._viewport_annotator = None

    def capture(self, obs_dict: dict, env=None):
        panels = []
        for cam in ("front", "wrist"):
            if cam in obs_dict:
                panels.append(obs_dict[cam][0].cpu().numpy().astype(np.uint8))
        if panels:
            if len(panels) == 2 and panels[0].shape[0] != panels[1].shape[0]:
                h = panels[0].shape[0]
                from PIL import Image
                p1 = np.array(Image.fromarray(panels[1]).resize(
                    (int(panels[1].shape[1] * h / panels[1].shape[0]), h)))
                panels[1] = p1
            self._robot_frames.append(np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0])

        if self._viewport_annotator is not None:
            try:
                data = self._viewport_annotator.get_data()
                if data is not None and hasattr(data, 'shape') and len(data.shape) >= 3:
                    self._scene_frames.append(np.array(data[:, :, :3], dtype=np.uint8))
            except Exception:
                pass

    def save(self, episode_id: int):
        import imageio
        if self._robot_frames:
            path = os.path.join(self.video_dir, f"episode_{episode_id}_robot.mp4")
            imageio.mimwrite(path, self._robot_frames, fps=self.fps, quality=8)
            print(f"[Video] Robot cams: {len(self._robot_frames)} frames -> {path}")
        if self._scene_frames:
            path = os.path.join(self.video_dir, f"episode_{episode_id}_scene.mp4")
            imageio.mimwrite(path, self._scene_frames, fps=self.fps, quality=8)
            print(f"[Video] Scene view: {len(self._scene_frames)} frames -> {path}")
        self._robot_frames.clear()
        self._scene_frames.clear()

    def reset(self):
        self._robot_frames.clear()
        self._scene_frames.clear()


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif args_cli.policy_type == "gr00tn1.6":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00t16ServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00t16ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )

    elif "lerobot" in args_cli.policy_type:
        from isaaclab.sensors import Camera
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"

        policy_type = args_cli.policy_type.split("-")[1]
        policy_dev = args_cli.policy_device or args_cli.device
        cam_remap = {}
        if args_cli.policy_camera_remap:
            for pair in args_cli.policy_camera_remap.split(","):
                src, dst = pair.split("=")
                cam_remap[src.strip()] = dst.strip()
        raw_cams = {
            key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)
        }
        camera_infos = {cam_remap.get(k, k): v for k, v in raw_cams.items()}
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos=camera_infos,
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=policy_dev,
        )
        if cam_remap:
            policy.set_camera_env_keys(list(raw_cams.keys()))
    elif args_cli.policy_type == "openpi":
        from isaaclab.sensors import Camera
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()
    recorder = None
    if args_cli.save_video:
        recorder = EpisodeVideoRecorder(args_cli.video_dir, fps=args_cli.step_hz)
        recorder.setup_viewport()
    step_logger = EpisodeStepLogger(args_cli.video_dir)

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    success_count, episode_count = 0, 1

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        if recorder:
            recorder.reset()
        step_logger.begin_episode(episode_count)
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    episode_count += 1
                    break

                obs_dict = preprocess_obs_dict(obs_dict["policy"], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    step_obs = obs_dict.get("policy", obs_dict)
                    step_logger.log_step(step_obs, action)
                    if recorder:
                        recorder.capture(step_obs, env)
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                step_logger.end_episode()
                if recorder:
                    recorder.save(episode_count)
                episode_count += 1
                success_count += 1
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                step_logger.end_episode()
                if recorder:
                    recorder.save(episode_count)
                episode_count += 1
                break
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
