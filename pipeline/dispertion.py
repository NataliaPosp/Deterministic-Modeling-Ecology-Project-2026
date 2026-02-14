import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def dispersion_analysis(a,m,d1,d2):
    delta = a**2 - 4*m**2
    if delta <= 0:
        st.warning("Delta <= 0: Brak stabilnego punktu wegetacji dla tych parametrów.")
        return None

    v_star = (a + np.sqrt(delta)) / (2*m)
    u_star = m / v_star

    fu = -1-v_star**2
    fv = -2*u_star*v_star
    gu = v_star**2
    gv = 2*u_star*v_star - m

    k_vals = np.linspace(0,3,200)
    lambdas = []

    for k in k_vals:
        tr_M = (fu + gv) - k**2 *(d1+d2)
        det_M = d1*d2*k**4 - (d1*gv + d2*fu)*k**2 + (fu*gv - fv*gu)

        term_sqrt = tr_M**2 - 4*det_M
        l_max = (tr_M + np.sqrt(term_sqrt + 0j))/2
        lambdas.append(l_max.real)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(k_vals, lambdas, color="darkorchid", label=r'$\lambda(k)$')
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.fill_between(k_vals, 0, lambdas, where=(np.array(lambdas) > 0), color="lime", alpha=0.3, label="Niestabilność Turinga")
    ax.set_xlabel("Liczba falowa $k$")
    ax.set_ylabel("$Re(\lambda)$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig