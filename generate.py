"""
Python script to generate a 2D LJ dataset
"""

import numpy as np
import torch
from ase import Atoms, io

# Settings
M = 10 # initial coordinate grid spacing
N = M**2 # Num atoms
L = 20.0 # Box length
dt = 0.01 # timestep
kT = 1.0 # KbT
gamma = 1.0 # Friction
sigma = 1.0 # LJ sigma
epsilon = 2.0 #  LJ epsilon
Nsteps = 100000 # Number of timesteps
Nout = 100 # Frequency of output

def energy(positions):
    "Compute the energy of a Lennard-Jones system."
   
    # get [i,j] pairs   
    a, b = torch.triu_indices(N, N, 1)
    
    # compute the displacement vectors
    d = positions[a] - positions[b]
    
    # PBCs
    d = d - torch.round(d/L)*L

    # compute distance^2
    r2 = torch.sum(d**2, dim=1)
    
    c6 = (sigma**2 / r2)**3
    c12 = c6**2
    
    return torch.sum(4 * epsilon * (c12 - c6))
    

# Simulation setup
line = torch.linspace(0,L-2.0,M)
x = torch.cartesian_prod(line,line)
v = torch.zeros(N,2)

# Main simulation loop
frames=[]
equilibrium_frames=[]
for t in range(Nsteps):
    
    x = x.clone().detach().requires_grad_(True)

    # compute energy
    energies = energy(x)

    # compute forces
    energies.backward()
    forces = -x.grad
 
    # leap-frog Verlet langevin
    alpha = np.exp(-gamma*dt)
    v = v*alpha + forces*(1.0-alpha)/gamma + np.sqrt(kT*(1-alpha**2))*torch.randn(v.shape)
    x = x.detach() + v*dt
    
    # PBCs
    x = x - torch.floor(x/L)*L


    # print some logs to screen
    if t%10000==0:
        print(t,"/",Nsteps, "Energy =", energies.item() )
    
    # output to trajectory
    if t%Nout==0:
        
        xyz = torch.hstack((x, torch.zeros(N,1)))
        frame = Atoms(''.join(['C']*N), positions = xyz.numpy(), cell=(L,L,L), pbc=True, info={"energy": float(energies.detach().numpy())})
        frames.append(frame)

        # save these to equilibrium frames
        equilibrium_frames.append(frame)

        ## add some noise to the data to get higher energy structures
        x_temp = x.clone()
        x_temp += torch.randn(x_temp.shape)*0.05
        e_temp = energy(x_temp)
        xyz = torch.hstack((x_temp, torch.zeros(N,1)))
        frame = Atoms(''.join(['C']*N), positions = xyz.numpy(), cell=(L,L,L), pbc=True, info={"energy": float(e_temp.detach().numpy())})
        frames.append(frame)


# write trajectorys using ASE
io.write("generated_traj_2d.extxyz", frames)
io.write("equilibrium_traj_2d.xyz", equilibrium_frames)