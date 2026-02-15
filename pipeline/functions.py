import streamlit as st
from pipeline.solver import KlausmeierSolver
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "patch_data.json")

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

    k_vals = np.linspace(0,10,200)
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

@st.cache_data
def pattern_plotting(a,m,d1,d2,Nx,Ny,Lx,Ly,ht):
    s = KlausmeierSolver(Nx, Ny, Lx, Ly, ht)
    v_star = (a + np.sqrt(a ** 2 - 4 * m ** 2)) / (2 * m)
    u = m / v_star * np.ones(Nx * Ny)
    v = v_star * np.ones(Nx * Ny) + 0.8 * np.random.randn(Nx * Ny)
    A_u, A_v = s.evolution_matrix(d1, d2)

    iter_count = 0
    while iter_count < 10000:
        iter_count += 1
        u_next, v_next = s.solve_step(u, v, a, m, A_u, A_v)
        u, v = u_next, v_next

        # Podgląd postępu
        if iter_count % 1000 == 0:
            print(f"Krok {iter_count}: sprawdzanie formowania wzorców...")

    fig, ax = plt.subplots()
    im = ax.imshow(v.reshape((Nx, Ny)), extent=[0, Lx, 0, Ly], origin='lower')
    fig.colorbar(im, ax=ax,label="Gęstość biomasy v")
    ax.set_title(f"Wzór dla a={a}, m={m}, d1={d1}, d2={d2}")

    variance = np.var(v)
    return fig, variance

@st.cache_data
def patch_size_analysis(a_list,m,d1,d2):
    """
    Funkcja analizująca wpływ rozmiaru środowiska na istnienie i stabilność rozwiązań.

    """
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r") as f:
                cached_res = json.load(f)
            if all(str(a) in cached_res for a in a_list):
                print("Wczytano dane z pliku!")
                return cached_res
        except Exception as e:
            print(f'Błąd wczytu: {e}')

    domain_sizes = np.linspace(10,100,10)
    all_means = {}
    Nx, Ny = 64, 64

    for a in a_list:
        means = []
        for L in tqdm.tqdm(domain_sizes):
            s = KlausmeierSolver(Nx, Ny, L, L, 0.005)
            v_star = (a + np.sqrt(a ** 2 - 4 * m ** 2)) / (2 * m)
            u = m / v_star * np.ones(Nx * Ny)
            v = v_star * np.ones(Nx * Ny) + 0.8 * np.random.randn(Nx * Ny)
            A_u, A_v = s.evolution_matrix(d1, d2)

            iter_count = 0
            v_prev = np.copy(v)
            while iter_count < 10000:
                iter_count += 1
                u, v = s.solve_step(u, v, a, m, A_u, A_v)
                if iter_count % 100 == 0:
                    diff = np.linalg.norm(v - v_prev) / np.linalg.norm(v)
                    if diff < 1e-4:  # Stan stacjonarny osiągnięty
                        break
                    v_prev = np.copy(v)
            means.append(np.mean(v))
            print(f"Zakończono dla L={L}")

        all_means[str(a)] = {"sizes":domain_sizes.tolist(),
                             "means": means}

        with open(DATA_PATH, 'w') as f:
            json.dump(all_means, f)

    return all_means


