
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.solver import KlausmeierSolver

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st


st.header("Analiza bifurkacyjna i tipping points")
st.markdown("Zaczniemy analizę od ustalenia parametrów. \n Niech $N_x, N_y = 41$, $L_x, L_y = 10$, $h_t=0.005$, $m=0.45$, $d_1=1.0$ oraz $d_2=0.01$")

st.write("Wykres po lewej przedstawia maksymalną wartość rozwiązania v w zależności od ilości "
         "opadów a. Wraz ze zmniejszającym się a (wielkością opadów), spada również maksimum z wyliczonych numerycznie opadów."
         " W pobliżu granicy wyznaczonej w analizie teoretycznej wynoszącej a=2m obserwujemy nagłą zmianę tendencji, skok i następnie "
         " punkt krytyczny (tipping point), w którym system gwałtownie skacze"
         " między stanem wegetacji a pustynią. Możemy zauważyć, że dla przyjętych parametrów iz niewymuszonym sztucznie punktem startowym na środku "
         " pola (w naszym wypadku wyszukiwanym numerycznie) linia powrotna nie zdołała się podnieść - osiągnięcie stanu pustynnego było stanem ostatecznym. "
         "")
st.info("Poniżej wygenerują się dwa wykresy, które będą podstawą analizy bifurkacji. Czas wykonywania symulacji to około 2 minut.")

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

