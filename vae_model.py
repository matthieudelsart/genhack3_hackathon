import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, input_dims):  
        super(VariationalEncoder, self).__init__()

        self.latent_dims = min(latent_dims, 50)

        self.fc1 = nn.Linear(input_dims, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, self.latent_dims)
        self.fc6 = nn.Linear(64, self.latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mu = self.fc5(x)
        sigma = torch.exp(self.fc6(x))
        N = self.N.sample(mu.shape)
        z = mu + sigma * N
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalDecoder(nn.Module):
    def __init__(self, latent_dims, output_dims):
        super(VariationalDecoder, self).__init__()

        self.latent_dims = min(latent_dims, 50)
        
        self.fc1 = nn.Linear(latent_dims, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, output_dims)
        
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_dims, output_dims, verbose):
        super(VariationalAutoencoder, self).__init__()
        self.verbose = verbose
        self.encoder = VariationalEncoder(latent_dims, input_dims)
        self.decoder = VariationalDecoder(latent_dims, output_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
### Training function
def train_vae(vae, X_train, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    batch = 25
    train_loss = 0.0
    verbose = vae.verbose
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i in range(0, len(X_train), batch):
        batch_X = X_train[i:i+batch].float()
        
        x_hat = vae(batch_X)

        # Evaluate loss
        loss = ((batch_X - x_hat)**2).sum() + vae.encoder.kl
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0 and verbose:
            # Print batch loss
            print('[%i] \t partial train loss (single batch): %f' % (i, loss.item()))

        train_loss += loss.item()

    return train_loss / len(X_train)


### Testing function
def test_vae(vae, X_test):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0

    with torch.no_grad(): # No need to track the gradients
        for i in range(len(X_test)):
            x = X_test[i].clone().detach().float()
            
            # Decode data
            x_test = x.unsqueeze(0)
            x_hat = vae(x_test)

            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(X_test)


yields_df = pd.read_csv('CSVs/yields_subset.csv')
yields_tensor = torch.tensor(yields_df.iloc[:, 2:].values)

verbose = True
epochs = 300
lr = 1e-3
latent_dims = 25

vae = VariationalAutoencoder(latent_dims=latent_dims, input_dims=4, output_dims=4, verbose=verbose)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr) #, weight_decay=1e-5)

# Train
# ----------------------------------------------------------
for epoch in range(epochs):
    train_loss = train_vae(vae,yields_tensor,optimizer)
    torch.cuda.empty_cache()
    if epoch % 10 == 0 and verbose:
        print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, epochs,train_loss))

# Import noise array
noise = np.load('data/noise.npy')[:, :latent_dims]
indx_range = np.arange(0, len(noise))
indx_selected = np.random.choice(indx_range, size=1000, replace=False)
noise = torch.from_numpy(noise[indx_selected])

# Load the model
generator = vae.decoder
generator.eval()

# Generate the distribution
yields_gen_tensor = generator(noise)
yields_gen_numpy = yields_gen_tensor.detach().numpy()

yields_gen_df = pd.DataFrame(yields_gen_numpy, columns=["YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"])

# Save the DataFrame to a CSV file
yields_gen_df.to_csv('CSVs/vae_yields_subset.csv', index=False)