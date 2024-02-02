import pandas as pd
import numpy as np
import torch

from models.gan_model import Generator, Discriminator, train_gan

# TRAIN MODEL
yields_df = pd.read_csv('CSVs/yields_subset.csv')
train_gan(yields_df, True)

# Import noise array
noise = np.load('data/noise.npy')
indx_range = np.arange(0, len(noise))
indx_selected = np.random.choice(indx_range, size=1000, replace=False)
noise = torch.from_numpy(noise[indx_selected])

# Load the model
generator = Generator(vector_shape=4)

generator_state_dict = torch.load("models/gan1_gen_model.pth")

generator.load_state_dict(generator_state_dict)

generator.eval()

# Generate the distribution
yields_gen_tensor = generator(noise)
yields_gen_numpy = yields_gen_tensor.detach().numpy()

yields_gen_df = pd.DataFrame(yields_gen_numpy, columns=[
                             "YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"])

# Save the DataFrame to a CSV file
yields_gen_df.to_csv('CSVs/gen_yields_subset.csv', index=False)
