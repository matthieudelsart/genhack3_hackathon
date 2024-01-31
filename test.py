from model import generative_model
import numpy as np

noise = np.load("data/noise.npy")
output = generative_model(noise)