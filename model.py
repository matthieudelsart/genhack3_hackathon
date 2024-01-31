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
from vae_model import VariationalAutoencoder
import torch

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

    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    full_model = VariationalAutoencoder(latent_dims=latent_dims, input_dims=4, output_dims=4, verbose=False)
    full_model.load_state_dict(torch.load('models/vae_model.pth'))
    
    model = full_model.decoder
    model.eval()

    return model(latent_variable) # G(Z)




