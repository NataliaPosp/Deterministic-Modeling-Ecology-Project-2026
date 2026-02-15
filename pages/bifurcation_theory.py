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

###

st.title("Ubezwymiarowienie - wyprowadzenie teoretyczne")
st.markdown("""
### Równania wyjściowe (Wymiarowe)
Analizujemy następujący wymiarowy model Klausmeiera-Graya-Scotta:
""")

st.latex(r'''
\begin{aligned}
\frac{\partial W}{\partial T} &= A - LW - RWN^2 + D_W \Delta W \quad \text{(Woda)} \\
\frac{\partial N}{\partial T} &= JRWN^2 - MN + D_N \Delta N \quad \text{(Biomasa)}
\end{aligned}
''')

st.markdown("""
### Wprowadzamy bezwymiarowe zmienne $u, v, t$ oraz współrzędną przestrzenną $x$:
""")

st.latex(r'''
W = W_0 u, \quad N = N_0 v, \quad T = T_0 t, \quad X = X_0 x
''')

st.markdown("Podstawiamy te zależności do pierwszego równania (dla wody):")

st.latex(r'''
\frac{\partial (W_0 u)}{\partial (T_0 t)} = A - L(W_0 u) - R(W_0 u)(N_0 v)^2 + D_W \frac{\partial^2 (W_0 u)}{\partial (X_0 x)^2}
''')

st.markdown(r"Dzielimy obustronnie przez $W_0$ i mnożymy przez $T_0$, aby wyizolować $$\frac{\partial u}{\partial t}$$:")

st.latex(r'''
\frac{\partial u}{\partial t} = \frac{A T_0}{W_0} - (L T_0)u - (R T_0 N_0^2)uv^2 + \left( \frac{D_W T_0}{X_0^2} \right) \frac{\partial^2 u}{\partial x^2}
''')

st.markdown(r"""Aby otrzymać postać $\frac{\partial u}{\partial t} = a - u - uv^2 + d_1 \Delta u$, musimy narzucić następujące warunki:
""")

st.latex(r'''
\begin{cases}
L T_0 = 1 \implies T_0 = \frac{1}{L} \\
R T_0 N_0^2 = 1 \implies N_0^2 = \frac{1}{R T_0} = \frac{L}{R} \implies N_0 = \sqrt{\frac{L}{R}} \\
\frac{D_W T_0}{X_0^2} = d_1
\end{cases}
''')

st.markdown("Z powyższych wynika definicja parametru $a$:")

st.latex(r'''
a = \frac{A T_0}{W_0} = \frac{A}{L W_0}
''')

st.markdown("""Podstawiamy analogicznie do równania na $N$:""")

st.latex(r'''
\frac{N_0}{T_0} \frac{\partial v}{\partial t} = J R (W_0 u) (N_0 v)^2 - M (N_0 v) + D_N \frac{N_0}{X_0^2} \frac{\partial^2 v}{\partial x^2}
''')

st.markdown(r"Dzielimy przez $\frac{N_0}{T_0}$:")

st.latex(r'''
\frac{\partial v}{\partial t} = (J R W_0 T_0 N_0) uv^2 - (M T_0) v + \left( \frac{D_N T_0}{X_0^2} \right) \frac{\partial^2 v}{\partial x^2}
''')

st.markdown("Aby współczynnik przy $uv^2$ wynosił 1, wyznaczamy $W_0$:")

st.latex(r'''
J R W_0 \frac{1}{L} \sqrt{\frac{L}{R}} = 1 \implies W_0 = \frac{L}{J R \sqrt{L/R}} = \frac{\sqrt{L}}{J \sqrt{R}}
''')

st.markdown("Ostatecznie otrzymujemy parametry $m$ oraz $d_2$:")

st.latex(r'''
m = M T_0 = \frac{M}{L}, \quad d_2 = \frac{D_N T_0}{X_0^2}
''')

st.markdown("A parametr $a$ przyjmuje postać:")
st.latex(r'''
a = \frac{A}{L} \cdot \frac{J \sqrt{R}}{\sqrt{L}} = \frac{J A \sqrt{R}}{L \sqrt{L}}
''')

st.markdown("Stąd uzyskaliśmy postać parametrów:")
st.latex(r'''
m = \frac{M}{L}, \quad a = \frac{J A \sqrt{R}}{L \sqrt{L}}, \quad d_1 = \frac{D_W}{LX_0^2}, \quad d_2 = \frac{D_N}{LX_0^2}, 
''')


st.markdown("### Wyznaczenie skali przestrzennej $X_0$")
st.markdown(r"""Współczynnik dyfuzji wody $d_1$ jest zdefiniowany jako
$d_1 = \frac{D_W T_0}{X_0^2}$. 
Przyjmując standardowo $d_1 = 1.0$, otrzymujemy bezpośrednią postać $X_0$:
""")

st.latex(r"X_0 = \sqrt{\frac{D_W}{L \cdot d_1}} = \sqrt{\frac{D_W}{L}}")

st.markdown("W takim wypadku ostateczna postać parametrów to:")
st.latex(r'''
m = \frac{M}{L}, \quad a = \frac{J A \sqrt{R}}{L \sqrt{L}}, \quad d_1 = 1.0, \quad d_2 = \frac{D_N}{D_W}, 
''')
st.markdown("---")


