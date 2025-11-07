
#-----------------------------------------------------------------------------------------
# Trainign script for incremental learning variant.
# first train the agent on single task (timing task). Then train it on dual task. To make sure the dual-task variant has learned the timing task correctly
#-----------------------------------------------------------------------------------------------------------

import argparse

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_environments.singleagent_dualtask import OverCookedSingleAgentDualtask
from rl_environments.singleagent_dualtask_delayedreward import OverCookedSingleAgentDualtaskDelayedReward
from feature_extractors.custom_cnn import CustomCNN
from rl_environments.singleagent_oneoven_incrementallearning import OverCookedSingleAgentIncrementallearning


def main():
    parser = argparse.ArgumentParser(description="Train OverCookedSingleAgent with RecurrentPPO")
    parser.add_argument("--oven_duration", type=int, required=True, help="Oven duration for the environment")
    parser.add_argument("--max_timesteps", type=int, required=True, help="Maximum timesteps in one episode") #500
    parser.add_argument("--cnn_output_size", type=int, required=True, help="flattened output size of cnn") # 125
    parser.add_argument("--save_path_singletask", type=str, required=True, help="Path to save the single task (T) trained model")
    parser.add_argument("--save_path_dualtask", type=str, required=True, help="Path to save the dual task (T+N) trained model")
    parser.add_argument("--total_timesteps_singletask", type=int, required=True, default=500000, help="total timesteps for model training") #100000
    parser.add_argument("--total_timesteps_dualtask", type=int, required=True, default=500000,help="total timesteps for model training")  # 150000
    # parser.add_argument("--gamma", type=float, required=True, default=0.99, help="discount factor used by the model")
    parser.add_argument("--ent_coef", type=float, required=True, default=0.0, help="entropy coefficient used by the model. Higher values means more exploration") # 0.05

    args = parser.parse_args()

    # Initialize environment with user-specified oven_duration
    env_singletask = OverCookedSingleAgentIncrementallearning(oven_duration=8, max_timesteps=500)
    env_dualtask = OverCookedSingleAgentDualtask(oven_duration=8, max_timesteps=500)

    print(f"Oven Duration set to: {env_singletask.oven_duration}")

    # Define policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,  # custom CNN added to handle small grid size of 5x3
        features_extractor_kwargs=dict(features_dim=args.cnn_output_size, n_channels=6),  # number of channels are different than single-task variant (5)
    )

    # Initialize model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env_singletask,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1000,
        ent_coef =args.ent_coef
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=f"models/{args.save_path_singletask}/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Train single task (T) model
    total_timesteps = args.total_timesteps_singletask
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    print(f"Single task Model saved to: {args.save_path_singletask}")

    model = RecurrentPPO.load(args.save_path_singletask, env=env_dualtask)

    # reduce learning rate to ensure the agent does not unlearn the timing task
    model.learning_rate = 0.0001

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=f"models/{args.save_path_dualtask}/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Train dual task (T+N) model
    total_timesteps = args.total_timesteps_dualtask
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, )


if __name__ == "__main__":
    main()