"""
Diagnostic: Measure typical loss magnitudes for your trained model.
This helps calibrate GECO τ targets and initial λ values.
"""
import json
import warnings
import numpy as np
import torch
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from vae import VAE
from vae_obs import VAEObstacleBCE
from sim.panda import Panda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configs
with open('../model_params/panda_10k/20260117_225605704-runcmd.json') as f:
    config = json.load(f)
    if 'parsed_args' in config:
        config = config['parsed_args']

# Load normalization
dataset = RobotStateDataset('../data', train=0, train_data_name='free_space_100k_train.dat')
mean_train = dataset.get_mean_train()
std_train = dataset.get_std_train()

obs_dataset = RobotObstacleDataset('../data', train=0,
    train_data_name='collision_100k_train.dat', test_data_name='collision_10k_test.dat',
    free_space_train_name='free_space_100k_train.dat', free_space_test_name='free_space_10k_test.dat')
mean_obs = obs_dataset.get_mean_train()[0, 10:14]
std_obs = obs_dataset.get_std_train()[0, 10:14]

# Load models
model = VAE(config['input_dim'], config['latent_dim'], config['units_per_layer'], config['num_hidden_layers'])
checkpoint = torch.load('../model_params/panda_10k/model.ckpt-016000.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

model_obs = VAEObstacleBCE(config['input_dim'], config['latent_dim'], config['units_per_layer'], config['num_hidden_layers'])
checkpoint_obs = torch.load('../model_params/panda_10k/snapshots_obs/model.ckpt-015350-000230.pt', map_location=device)
model_obs.load_state_dict(checkpoint_obs['model_state_dict'])
model_obs.to(device).eval()
model_obs.load_state_dict(checkpoint['model_state_dict'], strict=False)

robot = Panda()
robot.to(device)

mean_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
std_t = torch.tensor(std_train, dtype=torch.float32).to(device)

# Sample 50 random scenarios and measure loss magnitudes
np.random.seed(42)
torch.manual_seed(42)

q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

goals = []
priors = []
collisions = []
z_norms = []

print("=" * 60)
print("DIAGNOSTIC: Measuring loss magnitudes for your model")
print("=" * 60)

for i in range(50):
    q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
    e_start = robot.FK(q_start.clone(), device, rad=True)
    q_target = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
    e_target = robot.FK(q_target.clone(), device, rad=True)

    # Sample obstacle between start and goal
    t_val = np.random.uniform(0.3, 0.7)
    e_s = e_start.cpu().numpy()[0]
    e_t = e_target.cpu().numpy()[0]
    obs_pos = e_s + t_val * (e_t - e_s)
    obs_raw = np.array([obs_pos[0], obs_pos[1], np.random.uniform(0.5, 1.0), np.random.uniform(0.05, 0.15)], dtype=np.float32)
    obs_norm = (obs_raw - mean_obs) / std_obs

    with torch.no_grad():
        x_start = torch.cat([q_start, e_start], dim=1)
        x_start_norm = (x_start - mean_t[:, :10]) / std_t[:, :10]
        z = model.encoder(x_start_norm)[0]

        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_t[:, :10] + mean_t[:, :10]
        e_decoded = x_decoded[:, 7:10]

        L_goal = torch.norm(e_decoded - e_target).item()
        L_prior = (0.5 * torch.sum(z ** 2)).item()
        z_norm = torch.norm(z).item()

        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0).to(device)
        logit = model_obs.obstacle_collision_classifier(z, obs_tensor)
        p_coll = torch.sigmoid(logit).item()
        L_collision = (-np.log(1 - p_coll + 1e-8))

    goals.append(L_goal)
    priors.append(L_prior)
    collisions.append(L_collision)
    z_norms.append(z_norm)

print(f"\nOver 50 random scenarios at step 0:")
print(f"  L_goal:      mean={np.mean(goals):.4f}, std={np.std(goals):.4f}, range=[{np.min(goals):.4f}, {np.max(goals):.4f}]")
print(f"  L_prior:     mean={np.mean(priors):.4f}, std={np.std(priors):.4f}, range=[{np.min(priors):.4f}, {np.max(priors):.4f}]")
print(f"  L_collision: mean={np.mean(collisions):.4f}, std={np.std(collisions):.4f}, range=[{np.min(collisions):.4f}, {np.max(collisions):.4f}]")
print(f"  ||z||:       mean={np.mean(z_norms):.4f}, std={np.std(z_norms):.4f}")

print(f"\n  L_prior / L_goal ratio: {np.mean(priors)/np.mean(goals):.1f}x")
print(f"  L_collision / L_goal ratio: {np.mean(collisions)/np.mean(goals):.1f}x")

print(f"\n--- Recommended calibration ---")
# For GECO to work, τ should be set near the typical loss values
# so that the constraint violation C = L - τ ≈ 0 at equilibrium
print(f"  Suggested τ_prior_goal:  {np.mean(priors):.2f} (mean L_prior)")
print(f"  Suggested τ_obs_goal:    {np.mean(collisions):.2f} (mean L_collision)")
print(f"\n  To make λ*L_prior ≈ L_goal at start, use λ_prior ≈ {np.mean(goals)/np.mean(priors):.4f}")
print(f"  To make λ*L_collision ≈ L_goal at start, use λ_collision ≈ {np.mean(goals)/np.mean(collisions):.4f}")
print("=" * 60)
