#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z |-> G_\theta(Z)
############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_dims = 40
    latent_variable = noise[:, :latent_dims]  # choose the appropriate latent dimension of your model
    latent_variable = torch.from_numpy(latent_variable)

    # VAE CLASS
    class VariationalEncoder(nn.Module):
        def __init__(self, latent_dims, input_dims):  
            super(VariationalEncoder, self).__init__()

            self.latent_dims = min(latent_dims, 50)

            self.fc1 = nn.Linear(input_dims, 32)
            self.fc2 = nn.Linear(32, 128)
            self.fc3 = nn.Linear(128, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 64)
            self.fc6 = nn.Linear(64, self.latent_dims)
            self.fc7 = nn.Linear(64, self.latent_dims)

            self.N = torch.distributions.Normal(0, 1) # Try a prior which is a mixture of gaussians?
            self.kl = 0

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            mu = self.fc6(x)
            sigma = torch.exp(self.fc7(x))
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
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, 32)
            self.fc6 = nn.Linear(32, output_dims)
            
        def forward(self, z):
            z = F.relu(self.fc1(z))
            z = F.relu(self.fc2(z))
            z = F.relu(self.fc3(z))
            z = F.relu(self.fc4(z))
            z = F.relu(self.fc5(z))
            z = F.relu(self.fc6(z))
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

    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    full_model = VariationalAutoencoder(latent_dims=latent_dims, input_dims=4, output_dims=4, verbose=False)
    full_model.load_state_dict(torch.load('parameters/vae_model.pth'))

    model = full_model.decoder
    model.eval()

    return model(latent_variable).detach().numpy() # G(Z)




