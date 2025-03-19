from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

# Disorder strength
W = 0.0  # Adjust W as needed
delta=0.7     # site energy
t=-1.0        # spin-independent first-neighbor hop
theta = 30.0
soc=0.06      # spin-dependent second-neighbor hop
rashba=0.05   # spin-flip first-neighbor hop
soc_list=np.array([-0.06,-0.24]) # spin-dependent second-neighbor hop
zoc = 0.3*soc


def set_model(t, soc, rashba, delta):
    # Set up Kane-Mele model
    lat = [[1, 0, 0], [0.5,np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    
    model = tb_model(3, 3, lat, orb, nspin=2)
    
    # Generate disorder values for each site
    disorder_values = np.random.uniform(-W/2, W/2, size=len(orb))  
    
    # Set onsite energy with disorder
    onsite_energies = [delta + disorder_values[i] if i % 2 == 0 else -delta + disorder_values[i] for i in range(len(orb))]
    model.set_onsite(onsite_energies)
    
    # Definitions of Pauli matrices
    sigma_x = np.array([0., 1., 0., 0])
    sigma_y = np.array([0., 0., 1., 0])
    sigma_z = np.array([0., 0., 0., 1])
    r3h = np.sqrt(3.0)/2.0
    sigma_a = 0.5 * sigma_x - r3h * sigma_y
    sigma_b = 0.5 * sigma_x + r3h * sigma_y
    sigma_c = -1.0 * sigma_x

    # Spin-independent first-neighbor hops
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        model.set_hop(t, 0, 1, lvec)

    # Spin-dependent second-neighbor hops
    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 1, 1, lvec)

    model.set_hop(0.3 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])
    model.set_hop(-0.3 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])

    # Spin-flip first-neighbor hops
    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")

    return model


fig, ax = plt.subplots(1, 2, figsize=(7, 3))
nkr = 201
width = 10

for je, soc_val in enumerate(soc_list):
    my_model = set_model(t, soc_val, rashba, delta)
    # Potong model pada dua arah finite sehingga sisa periodisitas hanya pada arah 1 (010)
    # fin_model = my_model.cut_piece(1, fin_dir=2, glue_edgs=True)
    ribbon_model = my_model.cut_piece(width, fin_dir=1, glue_edgs=False)

    # Dengan model ribbon 1D (periodik hanya pada arah 1) kita bisa hitung band structure
    path=[[0.,0.],[2./3.,1./3.],[.5,.5],[1./3.,2./3.], [0.,0.]]
    label=(r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')
    (k_vec, k_dist, k_node) = ribbon_model.k_path( path, nkr, report=False)

    rib_eval, rib_evec = ribbon_model.solve_all(k_vec, eig_vectors=True)
    nbands = rib_eval.shape[0]

    ax1 = ax[je]
    ax1.set_xlim([0,k_node[-1]])
    ax1.set_xticks(k_node)
    ax1.set_xticklabels(label)
    ax1.set_ylim(-5, 5)
    ax1.set_ylabel("Edge band structure (010)")

    for i in range(len(k_vec)):
        # Karena periodik hanya pada arah 1 (010), gunakan dir=1 untuk ekspektasi posisi
        pos_exp = ribbon_model.position_expectation(rib_evec[:, i], dir=1)
        weight = np.minimum(3.0 * pos_exp / width, 1.0)
        # Gunakan k_dist sebagai x karena itu jarak sepanjang lintasan k-space 1D
        ax1.scatter([k_dist[i]] * nbands, rib_eval[:, i],
                    s=0.6 + 2.5 * weight, c='k', marker='o', edgecolors='r')

fig.tight_layout()
plt.savefig("kanemele_topo_bd_010.pdf")
plt.show()
