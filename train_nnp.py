import torch
import numpy as np
import ase.io

## radial basis settings
NB=16
RCUT=torch.tensor(5.0)
rb=torch.linspace(0.1, RCUT, NB)
w=torch.tensor(0.2*RCUT.item()/NB)

## number of features in the hidden layers
NF1 = 64
NF2 = 64

## torch settings
torch.set_default_dtype(torch.float32)
DTYPE = torch.get_default_dtype()

## training and validation split 
frac_train=0.9
frac_valid=0.1

## training settings
EPOCHS=100
BATCHSIZE=10

@torch.jit.script
def envelope(x):
    '''Polynomial envolope function'''
    return( 1 - 28*x**6 + 48*x**7 - 21*x**8 )

@torch.jit.script
def radial_basis(x,rb, w, RCUT):
    '''Radial basis functions'''

    f=envelope(x/RCUT)

    e = torch.exp(-(x[:,None] - rb[None,:])**2/w)*f[:,None]
    e = torch.sum(e, dim=0)
    return e


@torch.jit.script
def make_descriptor(x,rb,w, RCUT):
    '''Turns Tensor of coordinates into radial basis descriptors'''

    N_batch = x.shape[0]
    N_atoms = x.shape[1]

    batched = []
    for b in range(N_batch):
        descriptors = []

        # compute displacements
        d = x[b,:,None] - x[b,None,:]

        # remove self
        d = d[~torch.eye(N_atoms,dtype=torch.bool),:].reshape(N_atoms, N_atoms-1,2)

        # PBCs
        L=20.0 # Box length
        d = d - torch.round(d/L)*L

        distance = torch.sqrt(torch.sum(d**2, dim=-1))

        for i in range(N_atoms):

            Rb = radial_basis(distance[i,:], rb, w,  RCUT)

            descriptors.append(Rb)

        batched.append(torch.stack(descriptors))
    out = torch.stack(batched)
    return out



class NNP(torch.nn.Module):
    def __init__(self, NB, rb, w, RCUT):
        super().__init__()
        self.num_features = NB
        self.NF1 = NF1
        self.NF2 = NF2
        self.rb = rb
        self.w  = w
        self.RCUT = RCUT

        self.NN = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.NF1, dtype=torch.get_default_dtype()),
            torch.nn.SiLU(),
            torch.nn.Linear(self.NF1, self.NF2, dtype=torch.get_default_dtype()),
            torch.nn.SiLU(),
            torch.nn.Linear(self.NF2, 1, dtype=torch.get_default_dtype()),
        )

    def forward(self, x):

        # x is coords, turn into radial basis features
        descriptor = make_descriptor(x, self.rb, self.w, self.RCUT)

        Es = self.NN(descriptor)
   
        # total energy is the sum of the energy of each atom
        total_E = torch.sum(Es.squeeze(dim=-1), dim=-1)

        return total_E




if __name__ == "__main__":

    ## load in the data
    traj = [frame for frame in ase.io.iread("generated_traj_2d.extxyz")]
    data = []

    for frame in traj:
        coords = torch.tensor(frame.positions[:,:2], dtype=DTYPE) # Make 2D
        energy = torch.tensor(frame.info["energy"], dtype=DTYPE)
        data_record={"x":coords, "energy": energy}
        data.append(data_record)

    ## split
    train, valid = torch.utils.data.random_split(data,(frac_train,frac_valid))

    ## create the model
    model = NNP(NB, rb, w, RCUT)

    print(model)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())

    losses=[]
    valid_losses=[]
    for n in range(EPOCHS):

        idxs=np.arange(len(train))
        np.random.shuffle(idxs)
        batch_idxs = np.split(idxs, len(idxs)//BATCHSIZE)

        for b, batch_idx in enumerate(batch_idxs):

            batch = [train[i] for i in batch_idx]
            train_x = torch.stack([record["x"] for record in batch])
            train_label = torch.stack([record["energy"] for record in batch])
            
            # predict (forward pass)
            predict = model(train_x)

            # compute the loss
            loss = loss_fn(predict, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        valid_x = torch.stack([record["x"] for record in valid])
        valid_label = torch.stack([record["energy"] for record in valid])
        valid_predict = model(valid_x)
        valid_loss = loss_fn(valid_predict, valid_label)
        valid_error = torch.mean(torch.abs((valid_predict-valid_label)))
        print("Epoc ", n,"training loss: ", loss.item(), ", validation loss:", valid_loss.item(), "validation error:", valid_error.item() )
        losses.append(loss.detach().numpy())
        valid_losses.append(valid_loss.detach().numpy())

    # save the model
    torch.save(model, "model.pt")






