import os
import sys
import glob
import wandb
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    
    # Check power status before starting
    if not self._is_ac_power_connected():
        print("\n🛑 [CRITICAL] Laptop AC power is not connected! Training cannot start without AC power.")
        sys.exit(1)
        
    # Patch logger.log to check AC power after every iteration
    original_log = self.logger.log
    
    def patched_log(*args, **kwargs):
        original_log(*args, **kwargs)
        if not self._is_ac_power_connected():
            print("\n🛑 [CRITICAL] Laptop AC power disconnected! Killing training program to protect battery/prevent overheating.")
            sys.exit(1)
            
    self.logger.log = patched_log

    def _is_ac_power_connected(self) -> bool:
        # Check sysfs (Linux standard)
        for path in glob.glob("/sys/class/power_supply/*"):
            type_path = os.path.join(path, "type")
            status_path = os.path.join(path, "status")
            online_path = os.path.join(path, "online")
            
            # Check battery discharging status
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r") as f:
                        status = f.read().strip()
                    if status == "Discharging":
                        return False  # Definitely on battery
                except Exception:
                    pass
                    
            # Check AC Mains status
            if os.path.exists(type_path) and os.path.exists(online_path):
                try:
                    with open(type_path, "r") as f:
                        ps_type = f.read().strip()
                    if ps_type in ["Mains", "USB", "AC"]:
                        with open(online_path, "r") as f:
                            online = f.read().strip()
                        return online == "1"
                except Exception:
                    pass
    # Fallback to psutil if sysfs is not accessible
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is not None:
            return battery.power_plugged
    except Exception:
        pass
    # Default to True to avoid stopping training if status cannot be determined
    return True

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if self.logger.logger_type in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_velocity_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
