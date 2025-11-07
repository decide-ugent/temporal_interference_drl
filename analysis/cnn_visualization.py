import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np



cnn_layer = 4
channel = 3
# model_obj = lstm_outputcalculator

pi_cnn_activations = {}
vf_cnn_activations = {}
for cnn_t in model_obj.pi_cnn_layer_outputs:
    pi_cnn_activations[0] = pi_cnn_activations.get(0, []) + [cnn_t[0][0, :, :, :]]
    pi_cnn_activations[2] = pi_cnn_activations.get(2, []) + [cnn_t[2][0, :, :, :]]
    pi_cnn_activations[4] = pi_cnn_activations.get(4, []) + [cnn_t[4][0, :, :, :]]

for cnn_t in model_obj.vf_cnn_layer_outputs:
    vf_cnn_activations[0] = vf_cnn_activations.get(0, []) + [cnn_t[0][0, :, :, :]]
    vf_cnn_activations[2] = vf_cnn_activations.get(2, []) + [cnn_t[2][0, :, :, :]]
    vf_cnn_activations[4] = vf_cnn_activations.get(4, []) + [cnn_t[4][0, :, :, :]]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))  # Create three subplots

# Initialize images for both subplots
image_axes_1 = ax1.imshow(pi_cnn_activations[cnn_layer][0][channel])  # first cnn layer, first timestep, first channel
image_axes_2 = ax2.imshow(vf_cnn_activations[cnn_layer][0][channel])  # first cnn layer, first timestep, first channel
image_axes_3 = ax3.imshow(np.sum(model_obj.dataset_train[0][0].detach().numpy(), axis=2))  # 1st timestep, 1st batch,

ax1.set_title("Feature Map policy")
ax2.set_title("Feature Map value")
ax3.set_title("actual image")

plt.close()  # Prevent duplicate static display in Jupyter Notebook


# Function to update both images
def update_mask(j):
    image_axes_1.set_array(pi_cnn_activations[cnn_layer][j][channel])

    image_axes_2.set_array(vf_cnn_activations[cnn_layer][j][channel])
    image_axes_3.set_array(np.sum(model_obj.dataset_train[j][0].detach().numpy(), axis=2))
    ax3.set_title(f"Oven timer: {model_obj.oven_timer[j]}. Action: {model_obj.actions[j]}. Timestep: {j}")
    # print("reward", model_obj.rewards[j])
    # print("oven timer", model_obj.env.oven1_timer)
    return [image_axes_1, image_axes_2, image_axes_3]


# Create the animation
animated_hands = FuncAnimation(fig, update_mask, frames=range(len(model_obj.dataset_train)), blit=True)

# Display the animation
HTML(animated_hands.to_jshtml())