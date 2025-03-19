import kwant
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

# 1. Define the Kane-Mele model using PythTB (your existing code)
def set_model(t, soc, rashba, delta, W):
    # ... (your existing PythTB code here)
    return model

# 2. Convert PythTB model to Kwant system
def pythtb_to_kwant(pythtb_model):
    lat = kwant.lattice.general(pythtb_model._lat, pythtb_model._orb)
    syst = kwant.Builder()
    
    # Add onsite terms
    for site in pythtb_model._site_energies:
        i, energy = site
        syst[lat(i)] = energy  # Include disorder here if needed
    
    # Add hopping terms
    for hop in pythtb_model._hoppings:
        i, j, value = hop
        syst[lat(i), lat(j)] = value
    
    return syst

# 3. Attach leads and compute conductance
def compute_conductance(syst, energy):
    # Define leads (example: simple square lattice leads)
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0, 0)))
    lead[lat(0, 0, 0)] = 0  # Lead onsite term
    lead[lat.neighbors()] = -1  # Lead hopping
    
    # Attach leads to the system
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    # Finalize and compute conductance
    fsyst = syst.finalized()
    return kwant.smatrix(fsyst, energy).transmission(1, 0)

# 4. Main workflow
W = 0.1  # Disorder strength
pythtb_model = set_model(t=-1, soc=0.06, rashba=0.05, delta=0.7, W=W)
kwant_syst = pythtb_to_kwant(pythtb_model)
energy = 0.0  # Fermi level
conductance = compute_conductance(kwant_syst, energy)
print(f"Conductance at E={energy}: {conductance} e²/h")
