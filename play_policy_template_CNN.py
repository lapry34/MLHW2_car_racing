import tensorflow as tf
import numpy as np
import sys
from dataset_preprocessing import convert_IMG_numpy

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)

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

def play(env, model):

    seed = 2000
    obs, _ = env.reset(seed=seed)
    
    # drop initial frames
    action0 = np.array([0, 0.1, 0])
    for i in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        # scale the observation from 0-255 to 0-1              
        obs = convert_IMG_numpy(obs) 
        obs = obs / 255.0

        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor, axis=0)  # Add batch dimension
        p = model.predict(obs_tensor, verbose=0) # adapt to your model
        print("Predictions:", p)
        k = np.argmax(p)  # adapt to your model
        action = map_commands(k)
        print("Action:", action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated




env_arguments = {
    'domain_randomize': False,
    'continuous': True,
    'render_mode': 'human'
}

env_name = 'CarRacing-v3'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# your trained
model_path = "cnn_model.keras"
model = tf.keras.models.load_model(model_path)

play(env, model)


