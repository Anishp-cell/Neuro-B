import time
import torch
import numpy as np
from connectome_rl.src.connectome.graph_utils import load_circuit_data
from connectome_rl.src.envs.cpg_wrapper import CPGLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy
from connectome_rl.src.rl.ppo import PPOTrainer, PPOConfig
from connectome_rl.src.rl.buffer import RolloutBuffer

def test_training_speed():
    cd = load_circuit_data("connectome_rl/data/dna_circuit_tensors.pt")
    env = CPGLocomotionEnv(seed=42)
    obs_dim = 12
    act_dim = 4
    policy = ConnectomePolicy(obs_dim=obs_dim, act_dim=act_dim, circuit_data=cd)
    trainer = PPOTrainer(policy=policy, config=PPOConfig(learning_rate=3e-4))
    buffer = RolloutBuffer(buffer_size=200, obs_dim=obs_dim, act_dim=act_dim)

    obs, _ = env.reset(seed=42)
    t0 = time.time()
    for step in range(200):
        obs_t = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.no_grad():
            action, log_prob, _, val = policy.get_action_and_value(obs_t)
        act_np = action.squeeze(0).numpy()
        next_obs, reward, term, trunc, info = env.step(act_np)
        buffer.add(obs=obs, action=act_np, reward=reward, value=val.item(), log_prob=log_prob.item(), done=term or trunc)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    # Compute GAE and train 1 epoch
    buffer.compute_returns_and_advantages(last_value=0.0, last_done=False)
    metrics = trainer.train_step(buffer, batch_size=50, update_epochs=2)
    dt = time.time() - t0
    print(f"200 steps + PPO update completed in {dt:.2f} seconds!")
    print(f"Metrics: {metrics}")
    env.close()

if __name__ == "__main__":
    test_training_speed()
