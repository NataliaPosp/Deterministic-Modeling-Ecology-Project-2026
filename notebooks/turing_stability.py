import streamlit as st
import sys
import os
# Dodaje folder główny projektu do ścieżek wyszukiwania
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.solver import KlausmeierSolver

st.title("Analiza niestacjonarna Turinga")

st.markdown("""
### 1. Model i Punkty Równowagi
Analiza układu równań reakcji-dyfuzji:
""")

st.latex(r'''
\begin{cases} 
\frac{\partial u}{\partial t} = d_1 \Delta u + a - u - uv^2 \\
\frac{\partial v}{\partial t} = d_2 \Delta v + uv^2 - mv 
\end{cases}
''')

st.markdown("Szukamy punktów równowagi $(u^*, v^*)$ dla stanu stacjonarnego bez dyfuzji:")

st.latex(r'''
\begin{cases} 
a - u - uv^2 = 0 \\
v(uv - m) = 0 
\end{cases}
''')

st.markdown("#### Przypadek 1: Stan pustynny")
st.latex(r'''
1^\circ \quad (u^*, v^*) = (a, 0)
''')
st.info("Ten stan jest zawsze stabilny (przy braku dyfuzji).")

st.markdown("#### Przypadek 2: Stan wegetacji")
st.latex(r'''
2^\circ \quad uv - m = 0 \implies u = \frac{m}{v}
''')
st.markdown("Podstawiając do pierwszego równania:")
st.latex(r'''
a - \frac{m}{v} - \frac{m}{v}v^2 = 0 \implies -mv^2 + av - m = 0
''')
st.latex(r'''
\Delta = a^2 - 4m^2
''')

st.markdown("* Jeśli $a < 2m$ ($\Delta < 0$): brak rozwiązań rzeczywistych.")
st.markdown("* Jeśli $a = 2m$ ($\Delta = 0$): jedno rozwiązanie $v^* = \frac{a}{2m} = 1$.")
st.markdown("* Jeśli $a > 2m$ ($\Delta > 0$): dwa rozwiązania:")

st.latex(r'''
v^* = \frac{a \pm \sqrt{a^2 - 4m^2}}{2m}
''')

st.markdown("---")

st.markdown("### 2. Linearyzacja i Stabilność")
st.markdown("Macierz Jacobiego $J$ dla punktu $(u^*, v^*)$:")

st.latex(r'''
J = \begin{pmatrix} f_u & f_v \\ g_u & g_v \end{pmatrix} = 
\begin{pmatrix} -1 - v^2 & -2uv \\ v^2 & 2uv - m \end{pmatrix}
''')

st.markdown("Warunki stabilności (części rzeczywiste wartości własnych ujemne):")
st.latex(r'''
\begin{cases} 
\text{tr}(J) = f_u + g_v < 0 \\
\text{det}(J) = f_u g_v - f_v g_u > 0 
\end{cases}
''')

st.markdown("---")

st.markdown("### 3. Dołączenie Dyfuzji (Instabilność Turinga)")
st.markdown("Rozważamy zaburzenie w postaci falowej $e^{\lambda t} e^{ikx}$. Macierz układu z dyfuzją $H$:")

st.latex(r'''
H = J - k^2 D = \begin{pmatrix} f_u - k^2 d_1 & f_v \\ g_u & g_v - k^2 d_2 \end{pmatrix}
''')

st.markdown("Relacja dyspersji (równanie charakterystyczne):")
st.latex(r'''
\lambda^2 - \text{tr}(H)\lambda + \text{det}(H) = 0
''')

st.markdown("Gdzie:")
st.latex(r'''
\text{tr}(H) = (f_u + g_v) - k^2(d_1 + d_2)
''')
st.latex(r'''
\text{det}(H) = d_1 d_2 k^4 - (d_1 g_v + d_2 f_u) k^2 + \text{det}(J)
''')

st.markdown("Warunkiem powstania wzorów (struktur) jest $\text{det}(H) < 0$ dla pewnego $k^2$:")
st.latex(r'''
(d_1 g_v + d_2 f_u)^2 > 4 d_1 d_2 \text{det}(J)
''')

st.markdown("Zakres krytyczny liczb falowych $k$:")
st.latex(r'''
k^2 \in \left[ \frac{(d_1 g_v + d_2 f_u) \pm \sqrt{(d_1 g_v + d_2 f_u)^2 - 4 d_1 d_2 \text{det}(J)}}{2 d_1 d_2} \right]
''')