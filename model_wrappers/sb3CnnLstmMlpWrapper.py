import torch.nn as nn


# access intermediate layer outputs - cnnlstmmlp
class sb3CnnLstmMlpWrapper(nn.Module):
    def __init__(self, model):
        super(sb3CnnLstmMlpWrapper ,self).__init__()
        self.pi_cnn = model.policy.pi_features_extractor.cnn
        self.vf_cnn = model.policy.vf_features_extractor.cnn
        self.pi_features_extractor = model.policy.pi_features_extractor
        self.vf_features_extractor = model.policy.vf_features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.mlp_action_net = self.mlp_extractor.policy_net
        self.mlp_value_net = self.mlp_extractor.value_net
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net
        self.lstm_actor = model.policy.lstm_actor
        self.lstm_critic = model.policy.lstm_critic

    def forward(self ,x, x_value_rnn=None ,x_action_rnn=None, deterministic=True ):
        # cnn layer outputs
        x_pi = x.detach().clone().float().permute(0, 3, 1, 2)
        x_vf = x.detach().clone().float().permute(0, 3, 1, 2)
        pi_cnn_layer_outputs = {}
        vf_cnn_layer_outputs = {}

        for i, layer in enumerate(self.pi_cnn):
            x_pi = layer(x_pi)
            pi_cnn_layer_outputs[i] = x_pi.clone().detach().numpy()

        for i, layer in enumerate(self.vf_cnn):
            x_vf = layer(x_vf)
            vf_cnn_layer_outputs[i] = x_vf.clone().detach().numpy()

        # output of all layers
        cnn_policy_features = self.pi_features_extractor(x)
        cnn_value_features = self.vf_features_extractor(x)
        # x = self.mlp_extractor(x)
        x_action_rnn_output, x_action_rnn = self.lstm_actor(cnn_policy_features, x_action_rnn)
        x_value_rnn_output, x_value_rnn = self.lstm_critic(cnn_value_features, x_value_rnn)

        x_action_mlp = self.mlp_action_net(x_action_rnn_output)
        x_value_mlp = self.mlp_value_net(x_value_rnn_output)

        x_action = self.action_net(x_action_mlp)
        x_value = self.value_net(x_value_mlp)

        # distribution = model.policy._get_action_dist_from_latent(x_action_mlp)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)

        return pi_cnn_layer_outputs, cnn_policy_features, vf_cnn_layer_outputs, cnn_value_features, x_action_rnn_output, x_action_rnn, x_value_rnn_output, x_value_rnn, x_action_mlp, x_value_mlp, x_action, x_value