import numpy as np
import torch
from ase import Atoms, io
from train_mlp import make_descriptor, NNP

# Settings
M=10 # initial coordinate grid spacing
N=M**2 # Num atoms
L=20.0 # Box length
dt=0.01 # timestep
kT=1.0 # KbT
gamma=1.0 # Friction
# sigma = 1.0 # LJ sigma
#epsilon = 2.0 #  LJ epsilon
Nsteps = 10000 # Number of timesteps
Nout = 100 # Frequency of output


# Simulation setup
line = torch.linspace(0,L-2.0,M)
x = torch.cartesian_prod(line,line)
v = torch.zeros(N,2)

model = torch.load("model_backup.pt")
model.eval()

# Main simulation loop
frames=[]
for t in range(Nsteps):
    
    x = x.clone().detach()

    # unsqueeze to make a batch dimension
    
    x = torch.unsqueeze(x,0)
    # shape of x is [1, N_atom, 2]

    x = x.clone().detach().requires_grad_(True)
    # compute energy
    energies = model(x)

    # compute forces
    energies.backward()
    forces = -x.grad

    forces=forces.squeeze()
    x=x.detach().squeeze()
 
    # leap-frog Verlet langevin
    alpha = np.exp(-gamma*dt)
    v = v*alpha + forces*(1.0-alpha)/gamma + np.sqrt(kT*(1-alpha**2))*torch.randn(v.shape)
    
    #print(x.shape)
    x += v*dt
    
    # PBCs
    x = x - torch.floor(x/L)*L

    # print some logs to screen
    if t%1000==0:
        print(t,"/",Nsteps, "Energy =", energies.item() )

    # print data to trajectory
    if t%Nout==0:


        xyz = torch.hstack((x, torch.zeros(N,1)))
        frame = Atoms(''.join(['C']*N), positions = xyz.numpy(), cell=(L,L,L), pbc=True, info={"energy": float(energies.detach().numpy())})
        #frames.append(frame)

        # write trajectory using ASE
        if t==0:
            io.write("test_traj_2d.xyz", frame) 
        else:
            io.write("test_traj_2d.xyz", frame,append=True) 