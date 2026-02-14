
import sys
import os

# Dodaje folder główny projektu do ścieżek wyszukiwania
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.solver import KlausmeierSolver

import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sparse
from scipy.sparse.linalg import spsolve
import streamlit as st
import pandas as pd


st.header("Analiza bifurkacyjna i tipping points")
st.markdown("Zaczniemy analizę od ustalonych parametrów. Niech $N_x, N_y = 41$, $L_x, L_y = 10$, $h_t=0.005$, $m=0.45$, $d_1=1.0$ oraz $d_2=0.01$")

#Ustalone parametry
Nx, Ny = 41, 41
Lx, Ly = 10, 10
ht = 0.005
m = 0.45
d1 = 1.0
d2 = 0.05
a_vals = np.linspace(1.8,0.0,500)
a_vals_incr = a_vals[::-1]

@st.cache_data
def run_simulation():
    s = KlausmeierSolver(Nx,Nx,Lx,Ly,ht)
    v_max_desc, v_mean_desc, v_max_incr, v_mean_incr = s.solution_for_bifurcation(a_vals, m, d1, d2)
    tp = s.tip_point(a_vals, v_max_desc, v_max_incr)
    return v_max_desc, v_mean_desc, v_max_incr, v_mean_incr, tp

with st.spinner('Trwa symulacja...'):
    v_max_desc, v_mean_desc, v_max_incr, v_mean_incr, tp = run_simulation()

fig, axes = plt.subplots(1,2,figsize=(10,5))

axes[0].plot(a_vals, v_max_desc, 'o', label="a descending")
axes[0].plot(a_vals_incr, v_max_incr, 's', label="a increasing")
axes[0].axvline(x=tp, linestyle='--', label="tipping point")
axes[0].legend()
axes[0].set_xlabel("a")
axes[0].set_ylabel("max v")

axes[1].plot(a_vals, v_mean_desc, 'o', label="a descending")
axes[1].plot(a_vals_incr, v_mean_incr, 's', label="a increasing")
axes[1].legend()
axes[1].set_xlabel("a")
axes[1].set_ylabel("mean v")

fig.tight_layout()
st.pyplot(fig)

st.write("Wykres po lewej przedstawia maksymalną wartość rozwiązania v w zależności od ilości "
         "opadów a. Możemy zauważyć, że dla przyjętych parametrów ")