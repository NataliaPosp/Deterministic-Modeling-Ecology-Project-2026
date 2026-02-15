import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from pipeline.functions import patch_size_analysis, v_stationary
import numpy as np

st.header("Wpływ zmiany wielkości środowiska na rozwiązania")

st.markdown("W zależności od wybranego rozmiaru obszaru obserwujemy odmienne zachowanie dla małych rozmiarów. Tam, gdzie opady są większe, "
            " roślinności łatwiej przetrwać i średnia jest większa niż 0. Dla najmniejszej wielkości opadów uzyskaliśmy rozwiązania o bardzo niskiej średniej, "
            "co wskazuje na to, że na małych obszarach nie są w stanie tworzyć wzorów. Większe opady stwarzają lepsze warunki do tworzenia się wzorów, co obserwujemy "
            "w niezerowej średniej już dla małych rozmiarów obszaru (L>10). Inne obserwacje możemy poczynić w przypadku bardzo małych opadów, dla których dla L<15 mamy stale "
            " zerową średnią. Dla takich wartości opadów środowisko nie potrafi się odtworzyć. Należy wspomnieć, że te tendencje byłyby prawdopodobnie uwidocznione jeszcze bardziej, "
            "gdyby symulacja był przeprowadzona dla gęstszego wyboru podziału przedziału, z którego bierzemy wielkości domeny L. "
            "Wszystkie wielkości zaczynają się stabilizować dla wielkości krytycznej bliskiej L=64, od której zbiegają do asymptot w pobliżu poziomu rozważanego we "
            "wcześniejszej analizie stanu stacjonarnego. ")

a_list = [1.2,1.5,2.0,2.3]
m,d1,d2 = 0.45,80.0,0.01


results = patch_size_analysis(a_list,m,d1,d2)
colors = {"1.2":"pink","1.5":"darkorchid", "2.0":"royalblue", "2.3": "lime"}
fig, axes = plt.subplots(4,1, figsize=(10,10), sharex=True)
axes = axes.flatten()

for i, a in enumerate(a_list):
    ax = axes[i]
    res = results[str(a)]
    dom_sq = np.array(res["sizes"])**2
    means = res["means"]

    ax.plot(dom_sq, means, label=f'a={a}', color=colors[str(a)])

    nonzero = np.where(np.array(means) > 0.03)[0]
    if len(nonzero) > 0:
        idx = nonzero[0]
        critical_size = dom_sq[idx]
        ax.scatter([critical_size],[means[idx]], color="red", zorder=5)
        ax.annotate(f'$L_c \\approx {np.sqrt(int(critical_size))}$',
                    xy=(critical_size, means[idx]), xytext=(critical_size + 500, means[idx] - 0.5),
                    arrowprops=dict(arrowstyle="->", color='red'))

    v_s = v_stationary(a,m)
    ax.axhline(means[-1],color=colors[str(a)],linestyle="--", alpha=0.7, label=f'$v^* = {v_s:.2f}$')

    ax.set_xlabel(r"wielkość środowiska $L^2$")
    ax.set_ylabel(r"średnia wielkość biomasy $\mu_v$")
    ax.set_title(f"Zależność dla a={a}")
    ax.grid(True, alpha=0.3)

    ax.legend()
    plt.tight_layout()

st.pyplot(fig)