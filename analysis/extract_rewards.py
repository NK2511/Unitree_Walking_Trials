with open(r"Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/params/env.yaml") as f:
    lines = f.readlines()

in_rewards = False
current_name = None
rewards = {}

for l in lines:
    if l.startswith("rewards:"):
        in_rewards = True
        continue
    if in_rewards and (l.startswith("terminations:") or l.startswith("curriculum:")):
        break
    if in_rewards:
        if l.startswith("  ") and not l.startswith("    ") and ":" in l:
            current_name = l.strip().split(":")[0]
            rewards[current_name] = {}
        elif current_name and l.strip().startswith("weight:"):
            rewards[current_name]["weight"] = float(l.strip().split("weight:")[1].strip())
        elif current_name and l.strip().startswith("func:"):
            rewards[current_name]["func"] = l.strip().split("func:")[1].strip()

print(f"{'Reward Term':<26} | {'Weight':<10} | {'Function / Target'}")
print("-" * 75)
for k, v in rewards.items():
    fn = v.get("func", "").split("!")[-1].replace("python/name:", "").strip(" '")
    print(f"{k:<26} | {v.get('weight', 0.0):<10} | {fn}")
