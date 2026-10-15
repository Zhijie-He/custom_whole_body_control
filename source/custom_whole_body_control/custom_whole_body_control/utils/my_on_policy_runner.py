import os
from datetime import datetime

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)

        original_log = self.logger.log

        def log_with_timestamp(*args, **kwargs):
            # Print the original training log.
            original_log(*args, **kwargs)

            # Print an aligned timestamp followed by a blank line.
            pad = 40
            print(
                f"{'Log time:':>{pad}} {datetime.now():%Y-%m-%d %H:%M:%S}\n",
                flush=True,
            )

        self.logger.log = log_with_timestamp