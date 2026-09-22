CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config.unitree_g1_cfg import unitree_g1_footstep_env_cfg
from mdp.footstep_rewards import _patch_env_with_footstep_manager
from mjlab.envs import ManagerBasedRlEnv

try:
    cfg = unitree_g1_footstep_env_cfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg)
    _patch_env_with_footstep_manager(env)

    robot = env.scene["robot"]
    print("SUCCESS: Env initialized successfully!")
    print("Available robot.data properties:")
    for prop in sorted(dir(robot.data)):
        if not prop.startswith("_"):
            print(f"  {prop}")
except Exception as e:
    import traceback
    traceback.print_exc()
