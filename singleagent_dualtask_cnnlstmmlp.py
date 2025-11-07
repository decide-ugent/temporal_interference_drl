
#-----------------------------------------------------------------------------------------
# Training script for the dual task (T+N) variant.
# Only difference between training script of single task (T) is that the Number of channels on the observation space are different than single-task variant (5)
# This script can also be used to train dual-task delayed reward agent
#-----------------------------------------------------------------------------------------------------------

import argparse

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_environments.singleagent_dualtask import OverCookedSingleAgentDualtask
from rl_environments.singleagent_dualtask_delayedreward import OverCookedSingleAgentDualtaskDelayedReward
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
    env = OverCookedSingleAgentDualtask(oven_duration=args.oven_duration, max_timesteps=args.max_timesteps)
    print(f"Oven Duration set to: {env.oven_duration}")

    # Define policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,  # custom CNN added to handle small grid size of 5x3
        features_extractor_kwargs=dict(features_dim=args.cnn_output_size, n_channels=6),  # number of channels are different than single-task variant (5)
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

if __name__ == "__main__":
    main()