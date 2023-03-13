import MDAnalysis
import MDAnalysis.analysis.rdf
import MDAnalysis.transformations
import MDAnalysis.analysis.distances
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,3))
for fname, label in zip(["equilibrium_traj_2d.xyz", "test_traj_2d.xyz"],["reference","MLP"]):

    u = MDAnalysis.Universe(fname)

    # set the PBCs
    dim = [20.0, 20.0, 20.0, 90, 90, 90]
    transform = MDAnalysis.transformations.boxdimensions.set_dimensions(dim)
    u.trajectory.add_transformations(transform)

    # loop over the frames compute pair distribution functions
    hists=[]
    for ts in u.trajectory:
        dmat = MDAnalysis.analysis.distances.distance_array(u.atoms,u.atoms, box=u.dimensions)
        mask = np.eye(dmat.shape[0], dtype=bool)
        distances = dmat[~mask]

        bins, edges = np.histogram(distances, bins=100, range=(0,5.0))
        centers = 0.5*(edges[1:]+edges[:-1])
        hists.append(bins)

    r = centers
    hist = np.mean(np.array(hists), axis=0)/r

    plt.plot(r,hist,label=label)

plt.xlabel("r")
plt.ylabel("PDF(r)")
plt.legend()
plt.savefig("comparison.png")
