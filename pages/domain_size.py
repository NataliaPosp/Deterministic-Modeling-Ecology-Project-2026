import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from pipeline.functions import patch_size_analysis
import numpy as np

st.header("Wpływ zmiany wielkości środowiska na rozwiązania")
a_list = [1.5,2.0,2.3]
m,d1,d2 = 0.45,60.0,0.01
results = patch_size_analysis(a_list,m,d1,d2)
colors = {"1.5":"darkorchid", "2.0":"royalblue", "2.3": "lime"}
fig, ax = plt.subplots()
for a in a_list:
    res = results[str(a)]
    dom_sq = np.array(res["sizes"])**2
    means = res["means"]
    ax.plot(dom_sq, means, label=f'a={a}', color=colors[str(a)])

ax.set_xlabel(r"wielkość środowiska $L^2$")
ax.set_ylabel(r"średnia wielkość biomasy $\mu_v$")
ax.set_title("Zależność średniej biomasy od rozmiaru domeny")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)