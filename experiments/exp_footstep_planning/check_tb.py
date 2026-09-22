import os
from tensorboard.backend.event_processing import event_accumulator

log_file = "/home/nandhith/Python/Humanoid_Xterra_IITK/Angad_Footstep_Planning_Trials/logs/rsl_rl/angad_footstep/2026-07-16_11-44-07_Scratch_ClockReward_v1/events.out.tfevents.1784182452.nandhith-Dell-G15-5520.14238.0"

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

all_tags = sorted(ea.Tags()['scalars'])
print("=== OVERALL METRICS ===")
for tag in ["Train/mean_reward", "Train/mean_episode_length"]:
    if tag in all_tags:
        events = ea.Scalars(tag)
        print(f"{tag}:")
        for ev in events[-5:]:  # show last 5
            print(f"  Iter {ev.step:>4d} | {ev.value:>10.4f}")

print("\n=== EPISODE REWARD TERMS (Latest Iteration) ===")
ep_reward_tags = [t for t in all_tags if "Episode_Reward" in t]
for tag in sorted(ep_reward_tags):
    events = ea.Scalars(tag)
    last = events[-1]
    peak = max(events, key=lambda e: e.value)
    print(f"{tag:45s} | iter {last.step:4d} val = {last.value:9.4f}  (peak = {peak.value:9.4f} @ {peak.step})")

print("\n=== TERMINATIONS & METRICS (Latest Iteration) ===")
other_tags = [t for t in all_tags if "Episode_Termination" in t or "Metrics" in t or "Footstep" in t]
for tag in sorted(other_tags):
    events = ea.Scalars(tag)
    last = events[-1]
    print(f"{tag:45s} | iter {last.step:4d} val = {last.value:9.4f}")
