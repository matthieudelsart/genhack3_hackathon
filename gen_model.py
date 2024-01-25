import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Generator model
class Generator(nn.Module):
    def __init__(self, vector_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(50, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, vector_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
    
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, vector_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(vector_shape, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# Training loop
def train(generator, discriminator, train_loader, num_epochs, criterion, discriminator_optimizer, generator_optimizer, g_loss, d_loss, verbose):
    noise_path = 'data/noise.npy'
    noise_full = np.load(noise_path)
    
    for epoch in range(num_epochs):
        gen_losses = []
        disc_losses = []

        for data in train_loader:
            data = data[0].float()

            batch_size = 25
            indx_range = np.arange(0, len(noise_full))
            indx_selected = np.random.choice(indx_range, size=batch_size, replace=False)
            noise = torch.from_numpy(noise_full[indx_selected])

            # Generate fake images
            fake_data = generator(noise)

            # Train discriminator
            real_logits = discriminator(data)
            fake_logits = discriminator(fake_data.detach())

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            disc_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            fake_logits = discriminator(fake_data)

            gen_loss = criterion(fake_logits, real_labels)

            generator_optimizer.zero_grad()
            gen_loss.backward()
            generator_optimizer.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

        g_loss.append(np.mean(gen_losses))
        d_loss.append(np.mean(disc_losses))

        if verbose:
            print(f"Epoch {epoch + 1}, Gen Loss: {g_loss[-1]}, Disc Loss: {d_loss[-1]}")

def train_gan(train_data, verbose):
    vector_shape = 4

    # Split the DataFrame into input features (spectra) and labels
    yields = train_data.iloc[:, 2:].values

    # Convert the data to torch tensors
    yield_tensor = torch.from_numpy(yields).float()

    # Create a TensorDataset
    dataset = TensorDataset(yield_tensor)

    # Set batch size and number of workers
    batch_size = 25
    num_workers = 0

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    g_loss = []
    d_loss = []

    # Initialize generator and discriminator
    generator = Generator(vector_shape)
    discriminator = Discriminator(vector_shape)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

    # Train the model
    EPOCHS = 100
    print(f"Training the GAN model...")
    train(generator, discriminator, train_loader, EPOCHS, criterion, discriminator_optimizer, generator_optimizer, g_loss, d_loss, verbose)
    print("Done!")
    torch.save(generator.state_dict(), f"models/gan1_gen_model.pth")
    torch.save(discriminator.state_dict(), f"models/gan1_disc_model.pth")