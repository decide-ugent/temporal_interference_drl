import numpy as np
import torch as th

# get intermediate layer outputs
class LayerOutputCalculator():

    def __init__(self, wrapped_model, model, env, has_lstm=True):
        self.wrapped_model = wrapped_model
        self.model = model
        self.env = env
        self.has_lstm = has_lstm

    def get_layer_outputs_per_episode(self, first_obs, timesteps=200):

        obs, _ = self.env.reset(seed=16)

        while np.where(first_obs[:, :, 1] == 1) != np.where(obs[:, :, 1] == 1):
            obs, _ = self.env.reset(seed=16)

        visual_obs = np.sum(obs, axis=2)
        states = None
        self.rewards = []
        self.oven_timer = [self.env.oven1_timer]

        self.dataset_train = []

        self.values = []
        self.actions = []

        self.pi_cnn_layer_outputs = []
        self.vf_cnn_layer_outputs = []

        self.cnn_policy_features = []
        self.cnn_value_features = []

        self.mlp_policy_features = []
        self.mlp_value_features = []

        if self.has_lstm:
            self.lstm_states_actor_h = []
            self.lstm_states_actor_c = []
            self.lstm_state_critic_h = []
            self.lstm_state_critic_c = []
            self.lstm_value_output = []
            self.lstm_action_output = []
            self.lstm_hidden = []  # combined internal state of lstm for both policy and value
            self.lstm_cell = []  # combined internal state of lstm for both policy and value

        for i in range(timesteps):

            self.dataset_train.append(th.tensor(obs, dtype=th.float32).unsqueeze(0))

            if self.has_lstm:
                if len(self.lstm_state_critic_h) == 0:
                    pi_cnn_layer_outputs, cnn_policy_feature, vf_cnn_layer_outputs, cnn_value_feature, x_action_rnn_output, lstm_states_actor, x_value_rnn_output, lstm_states_critic, x_action_mlp, x_value_mlp, x_action, x_value = self.wrapped_model(
                        self.dataset_train[-1])
                else:
                    pi_cnn_layer_outputs, cnn_policy_feature, vf_cnn_layer_outputs, cnn_value_feature, x_action_rnn_output, lstm_states_actor, x_value_rnn_output, lstm_states_critic, x_action_mlp, x_value_mlp, x_action, x_value = self.wrapped_model(
                        self.dataset_train[-1], lstm_states_actor, lstm_states_critic)
            else:
                pi_cnn_layer_outputs, cnn_policy_feature, vf_cnn_layer_outputs, cnn_value_feature, x_action_mlp, x_value_mlp, x_action, x_value = self.wrapped_model(
                    self.dataset_train[-1])

            # actions2.append(action_probs)
            self.values.append(x_value.detach().numpy()[0])

            if self.has_lstm:
                self.lstm_states_actor_h.append(
                    lstm_states_actor[0].detach().numpy())  # internal states of lstm used in next timestep
                self.lstm_states_actor_c.append(lstm_states_actor[1].detach().numpy())
                self.lstm_state_critic_h.append(lstm_states_critic[0].detach().numpy())
                self.lstm_state_critic_c.append(lstm_states_critic[1].detach().numpy())
                self.lstm_value_output.append(x_value_rnn_output.detach().numpy())  # lstm output used by mlp layer
                self.lstm_action_output.append(x_action_rnn_output.detach().numpy())
            self.mlp_policy_features.append(x_action_mlp.detach().numpy())
            self.mlp_value_features.append(x_value_mlp.detach().numpy())
            self.cnn_policy_features.append(cnn_policy_feature.detach().numpy())
            self.cnn_value_features.append(cnn_value_feature.detach().numpy())
            self.pi_cnn_layer_outputs.append(pi_cnn_layer_outputs)
            self.vf_cnn_layer_outputs.append(vf_cnn_layer_outputs)

            if self.has_lstm:
                action, states = self.model.predict(obs, state=states, deterministic=True)
                self.lstm_hidden.append(states[0])  # combined policy and value
                self.lstm_cell.append(states[1])  # combined policy and value
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            self.actions.append(action)

            print(visual_obs)
            print("action", action)
            obs, reward, done, _, _ = self.env.step(action)
            self.oven_timer.append(self.env.oven1_timer)
            visual_obs = np.sum(obs, axis=2)

            print("reward", reward)
            print("timestep", self.env.timestep)
            print("oven timer", self.env.oven1_timer)
            self.rewards.append(reward)

            if done:
                print("total rewards", np.sum(self.rewards))
                break

