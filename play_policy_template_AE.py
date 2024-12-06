import joblib
import tensorflow as tf
import numpy as np
import sys
sys.path.append("./AE")
from train_AE import Encoder, Decoder, AutoEncoder

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)


class CustomModel():
    def __init__(self, AE_path, SK_path):
        self.ae = tf.keras.models.load_model(AE_path)
        self.encoder = self.ae.encoder
        self.sklearn_model = joblib.load(SK_path)

    
    def predict(self, obs_tensor):
        z = self.encoder(obs_tensor)
        z = z.numpy()
        return self.sklearn_model.predict(z)

def play(env, model):

    seed = 2000
    obs, _ = env.reset(seed=seed)
    
    # drop initial frames
    action0 = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    for i in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        # scale the observation from 0-255 to 0-1        
        obs = obs / 255.0

        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor, axis=0)  # Add batch dimension
        print("Observation shape:", obs_tensor.shape)
        p = model.predict(obs_tensor) # adapt to your model
        print("Predictions:", p)
        k = np.argmax(p)  # adapt to your model
        action = np.zeros(5, dtype=np.int32)
        action[k] = 1
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
AE_path = "./AE/autoencoder_4.keras"
SKLEARN_path = "./AE/svm_AE_4.joblib"
SKLEARN_path = "./AE/kNN_AE_4.joblib"
model = CustomModel(AE_path, SKLEARN_path)

play(env, model)


