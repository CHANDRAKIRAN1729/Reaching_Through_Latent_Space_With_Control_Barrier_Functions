"""
Analyze: Where along the path do collisions happen?
Check if collisions are concentrated in early steps.
"""
import json
import numpy as np

with open('../model_params/panda_10k/evaluation_results_detailed.json') as f:
    results = json.load(f)

detailed = results['detailed_results']

total = len(detailed)
goal_reached = sum(1 for r in detailed if r['goal_reached'])
collision_free = sum(1 for r in detailed if r['is_collision_free'])
success = sum(1 for r in detailed if r['success'])

# Analyze collision timing
early_only = 0  # collision ONLY in first 10% of steps
late_only = 0
both = 0
no_collision = 0
goal_but_collision = 0

for r in detailed:
    steps = r['collision_waypoint_steps']
    n_steps = r['num_steps']
    
    if len(steps) == 0:
        no_collision += 1
        continue
    
    threshold = max(1, int(n_steps * 0.1))  # first 10% of steps
    early = [s for s in steps if s < threshold]
    late = [s for s in steps if s >= threshold]
    
    if early and not late:
        early_only += 1
    elif late and not early:
        late_only += 1
    elif early and late:
        both += 1
    
    if r['goal_reached'] and not r['is_collision_free']:
        goal_but_collision += 1

print("=" * 60)
print("COLLISION TIMING ANALYSIS")
print("=" * 60)
print(f"Total scenarios:     {total}")
print(f"Goal reached:        {goal_reached} ({goal_reached/total*100:.1f}%)")
print(f"Collision-free:      {collision_free} ({collision_free/total*100:.1f}%)")
print(f"Success:             {success} ({success/total*100:.1f}%)")
print()
print(f"Goal reached but collision: {goal_but_collision}")
print(f"  → These paths WOULD be successes if not for collisions")
print()
print("Where do collisions occur?")
print(f"  No collision at all:   {no_collision}")
print(f"  Early only (first 10%): {early_only}")
print(f"  Late only:              {late_only}")
print(f"  Both early & late:      {both}")
print()

# What if we skip first N steps?
for skip in [5, 10, 20, 30, 50]:
    cf = 0
    succ = 0
    for r in detailed:
        steps = r['collision_waypoint_steps']
        late_collisions = [s for s in steps if s >= skip]
        is_cf = (len(late_collisions) == 0)
        if is_cf:
            cf += 1
        if r['goal_reached'] and is_cf:
            succ += 1
    print(f"  Skip first {skip:>3} steps → Collision-free: {cf/total*100:.1f}%, Success: {succ/total*100:.1f}%")

print("=" * 60)
