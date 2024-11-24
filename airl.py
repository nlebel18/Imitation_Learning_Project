import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial import GAIL
from imitation.util.util import make_vec_env

# Configuration de l'environnement
env_name = "CartPole-v1"
env = make_vec_env(env_name, n_envs=1)

# Politique experte pré-entraînée
expert = PPO("MlpPolicy", env, verbose=1)
expert.learn(5000)

# Données d'évaluation pour AIRL
reward_means = []
reward_stds = []
timesteps = []

# Initialisation de l'algorithme AIRL (remplacez GAIL par AIRL si disponible)
gail = GAIL(
    venv=env,
    expert_data=None,  # Ajoutez vos trajectoires expertes ici
    expert_policy=expert.policy,
    demo_batch_size=128,
)

# Entraînement et évaluation
total_timesteps = 10000
evaluation_interval = 1000

for t in range(0, total_timesteps, evaluation_interval):
    # Entraîner AIRL
    gail.train(evaluation_interval)
    
    # Évaluation de la politique
    mean_reward, std_reward = evaluate_policy(gail.gen_policy, env, n_eval_episodes=10)
    
    # Collecte des données pour le tracé
    reward_means.append(mean_reward)
    reward_stds.append(std_reward)
    timesteps.append(t + evaluation_interval)

# Tracé des résultats
plt.figure(figsize=(10, 6))
plt.plot(timesteps, reward_means, label="Mean Reward", marker="o")
plt.fill_between(
    timesteps,
    np.array(reward_means) - np.array(reward_stds),
    np.array(reward_means) + np.array(reward_stds),
    alpha=0.2,
    label="Std Reward",
)
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Performance of AIRL")
plt.legend()
plt.grid()
plt.show()
