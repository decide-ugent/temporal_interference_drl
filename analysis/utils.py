
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def neural_activations_by_timestep(policy_layer, value_layer, total_timestep=100):
    policy_output = np.array([h_t.flatten() for h_t in policy_layer])[:total_timestep, :]
    value_output = np.array([h_t.flatten() for h_t in value_layer])[:total_timestep, :]

    policy_l2norm = np.array([np.linalg.norm(h_t.flatten()) for h_t in policy_layer])[:total_timestep]
    value_l2norm = np.array([np.linalg.norm(h_t.flatten()) for h_t in value_layer])[:total_timestep]

    fig, axes = plt.subplots(2, 1, figsize=(25, 7))

    # Hidden state
    axes[0].plot(policy_output, marker="o", alpha=0.5)
    axes[1].plot(value_output, marker="o", alpha=0.5)
    # axes[0].set_xticks(range(0,policy_output.shape[0]))
    # axes[1].set_xticks(range(0,value_output.shape[0]))
    axes[0].set_xticks(range(0, total_timestep))
    axes[1].set_xticks(range(0, total_timestep))
    # Overlay L2 norm
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(policy_l2norm, color='black', linewidth=2, label='L2 Norm')
    ax0_twin.set_ylabel("L2 Norm", color='black')
    ax0_twin.tick_params(axis='y', labelcolor='black')

    ax1_twin = axes[1].twinx()
    ax1_twin.plot(value_l2norm, color='black', linewidth=2, label='L2 Norm')
    ax1_twin.set_ylabel("L2 Norm", color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black')

    plt.tight_layout()
    plt.show()

def pca(squeezed_policy_layer, squeezed_value_layer, pca_components=3, total_timestep=100):
    scaler_policy = StandardScaler()
    scaler_value = StandardScaler()

    policy_layer_scaled = scaler_policy.fit_transform(squeezed_policy_layer)
    value_layer_scaled = scaler_value.fit_transform(squeezed_value_layer)

    pca_policy = PCA(n_components=pca_components)
    pca_value = PCA(n_components=pca_components)

    policy_layer_pc = pca_policy.fit_transform(policy_layer_scaled)
    value_layer_pc = pca_value.fit_transform(value_layer_scaled)

    explained_variance_policy = pca_policy.explained_variance_ratio_
    explained_variance_value = pca_value.explained_variance_ratio_

    print("explained_variance policy", explained_variance_policy)
    print("explained_variance value", explained_variance_value)

    fig, axes = plt.subplots(2, 1, figsize=(25, 7))

    # Hidden state
    axes[0].plot(policy_layer_pc, marker="o" ,)
    axes[1].plot(value_layer_pc, marker="o" ,)
    # axes[0].set_xticks(range(0,policy_output.shape[0]))
    # axes[1].set_xticks(range(0,value_output.shape[0]))
    axes[0].set_xticks(range(0 ,total_timestep))
    axes[1].set_xticks(range(0 ,total_timestep))
    plt.xlabel("Time Step")
    plt.ylabel("Hidden State")
    plt.show()
    return policy_layer_pc, value_layer_pc


def fft(squeezed_policy_layer, squeezed_value_layer, ):
    # N - number of neurons
    # T - time steps
    squeezed_policy_layer_NT = squeezed_policy_layer.T  # Shape: (N, T)
    squeezed_value_layer_NT = squeezed_value_layer.T  # Shape: (N, T)

    # Apply FFT to each neuron's time series
    fft_policy_results = np.fft.fft(squeezed_policy_layer_NT, axis=1)
    fft_policy_magnitudes = np.abs(fft_policy_results)

    fft_value_results = np.fft.fft(squeezed_value_layer_NT, axis=1)
    fft_value_magnitudes = np.abs(fft_value_results)

    # Compute frequency bins
    time_steps = squeezed_policy_layer.shape[0]
    sampling_rate = 1  # 1 sample per timestep; change if you have a specific rate
    freqs = np.fft.fftfreq(time_steps, d=1 / sampling_rate)

    fig, axes = plt.subplots(2, 1, figsize=(25, 7))

    for i in range(squeezed_policy_layer.shape[1]):
        axes[0].plot(freqs[:time_steps // 2], fft_policy_magnitudes[i, :time_steps // 2], marker="o", alpha=0.5,
                     color="black", linestyle="--")
        axes[1].plot(freqs[:time_steps // 2], fft_value_magnitudes[i, :time_steps // 2], marker="o", alpha=0.5,
                     color="black", linestyle="--")
        # break

    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Magnitude", fontsize=12)
    plt.xticks(np.arange(0, 0.5, 0.05))
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.savefig("fft_of_lstm_hidden_states_changing_frame.png", dpi=300, bbox_inches="tight")
    plt.show()
