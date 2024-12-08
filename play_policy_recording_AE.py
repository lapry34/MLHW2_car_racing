import joblib
from matplotlib.pyplot import imshow
import tensorflow as tf
import numpy as np
import sys
import gymnasium as gym

sys.path.append("./AE")
from dataset_preprocessing import convert_IMG_numpy
from train_AE import Encoder, Decoder, AutoEncoder

def map_commands(argmax):
    # Initialize the continuous action array: [steering, gas, braking]
    action = np.zeros(3, dtype=np.float32)

    if argmax == 0:  # Do nothing
        action[:] = [0.0, 0.0, 0.0]
    elif argmax == 2:  # Steer left
        action[:] = [-1.0, 0.0, 0.0]
    elif argmax == 1:  # Steer right
        action[:] = [1.0, 0.0, 0.0]
    elif argmax == 3:  # Gas
        action[:] = [0.0, 1.0, 0.0]
    elif argmax == 4:  # Brake
        action[:] = [0.0, 0.0, 1.0]

    return action

class CustomModel():
    def __init__(self, AE_path, cl_path, keras_model=False):
        self.ae = tf.keras.models.load_model(AE_path)
        self.encoder = self.ae.encoder
        self.keras_model = keras_model
        if keras_model:
            self.classificator = tf.keras.models.load_model(cl_path)
        else:
            self.classificator = joblib.load(cl_path)

    def predict(self, obs_tensor):
        z = self.encoder(obs_tensor)
        z = z.numpy()
        return self.classificator.predict(z)

def play(env, model):
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # Drop initial frames
    action0 = np.array([0, 1, 0])
    for i in range(70):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        # Scale the observation from 0-255 to 0-1     
        obs = convert_IMG_numpy(obs)    
        obs = obs / 255.0

        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor, axis=0)  # Add batch dimension
        p = model.predict(obs_tensor)  # Adapt to your model
        if model.keras_model:
            p = np.argmax(p)
        action = map_commands(p)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env_arguments = {
    'domain_randomize': False,
    'continuous': True,
    'render_mode': 'rgb_array'  # Turn off rendering
}

env_name = 'CarRacing-v3'
video_folder = "./recordings"
env = gym.make(env_name, **env_arguments)

# Use the RecordVideo wrapper to save the recording
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, name_prefix="recording")

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Your trained model paths
AE_path = "./AE/autoencoder_4.keras"
cl_path = "./AE/kNN_AE_4.joblib"
model = CustomModel(AE_path, cl_path, keras_model=False)

play(env, model)

# Close the environment
env.close()

print(f"Recording saved in: {video_folder}")