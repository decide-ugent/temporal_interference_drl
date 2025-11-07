
#-----------------------------------------------------------------------------------------
# Training script for single task (T) variant.
# It also includes code to test the model
#-----------------------------------------------------------------------------------------------------------

import argparse

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_environments.singleagent_oneoven import OverCookedSingleAgentOneOven
from feature_extractors.custom_cnn import CustomCNN

def main():
    parser = argparse.ArgumentParser(description="Train OverCookedSingleAgent with RecurrentPPO")
    parser.add_argument("--oven_duration", type=int, required=True, help="Oven duration for the environment")
    parser.add_argument("--max_timesteps", type=int, required=True, help="Maximum timesteps in one episode") #500
    parser.add_argument("--cnn_output_size", type=int, required=True, help="flattened output size of cnn") # 125
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--total_timesteps", type=int, required=True, default=500000, help="total timesteps for model training") #200000
    # parser.add_argument("--gamma", type=float, required=True, default=0.99, help="discount factor used by the model")
    parser.add_argument("--ent_coef", type=float, required=True, default=0.0, help="entropy coefficient used by the model. Higher values means more exploration") # 0.05

    args = parser.parse_args()

    # Initialize environment with user-specified oven_duration
    env = OverCookedSingleAgentOneOven(oven_duration=args.oven_duration, max_timesteps=args.max_timesteps)
    print(f"Oven Duration set to: {env.oven_duration}")

    # Define policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,  # custom CNN added to handle small grid size of 5x3
        features_extractor_kwargs=dict(features_dim=args.cnn_output_size, n_channels=5),  # number of channels are different than dual-task variant (6)
    )

    # Initialize model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1000,
        ent_coef =args.ent_coef
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=f"models/{args.save_path}/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Train model
    total_timesteps = args.total_timesteps
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    print(f"Model saved to: {args.save_path}")

    print("Testing model")
    test_model(args.save_path)


def test_model(model_path):
    env_test = OverCookedSingleAgentOneOven(oven_duration=8)

    model = RecurrentPPO.load(f"models/{model_path}/rl_model_100000_steps.zip", env=env_test)
    # model = RecurrentPPO.load("models/overcooked_wait_singleagent_T8_v0_logs/rl_model_conttrain_98000_steps.zip", env=env_test)
    # print(f)
    obs, _ = env_test.reset()
    visual_obs = np.sum(obs, axis=2)
    # print("oven timers", env_test.oven1_timer, env_test.oven2_timer)
    rewards = 0
    states = None
    for step in range(5000):
        # obs = obs + 300
        action, states = model.predict(obs, state=states, deterministic=True)
        print(visual_obs)
        obs, reward, done, _, _ = env_test.step(action)
        visual_obs = np.sum(obs, axis=2)

        print(f"Step {step}: action={action}, reward={reward}")

        rewards += reward
        if done:
            obs, _ = env_test.reset()

            print("Episode ended. Total rewards", rewards)
            break


if __name__ == "__main__":
    main()
