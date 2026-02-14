import streamlit as st
import sys
import os
# Dodaje folder główny projektu do ścieżek wyszukiwania
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.solver import KlausmeierSolver
from pipeline.functions import dispersion_analysis

st.title("Analiza niestacjonarna Turinga")

st.markdown("""
### 1. Model i Punkty Równowagi
Analiza układu równań reakcji-dyfuzji:
""")

st.latex(r'''
\begin{cases} 
\frac{\partial u}{\partial t} =  a - u - uv^2 + d_1 \Delta u \\
\frac{\partial v}{\partial t} =  uv^2 - mv + d_2 \Delta v \\
u = v = 0, x \in \partial \Omega, t > 0 \\
u(x,0)=u_0(x), v(x,0)=v_0(x), x \in \Omega
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
st.markdown(r"Rozważamy przypadek $v^* \neq 0$.")
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

st.markdown("* Jeśli $a < 2m$ ($\Delta < 0$): brak rozwiązań rzeczywistych (stanów równowagi).")
st.markdown(r"* Jeśli $a = 2m$ ($\Delta = 0$): jedno rozwiązanie $v^* = \frac{a}{2m} = 1$, ale szukamy punktu v^* > 1")
st.markdown("* Jeśli $a > 2m$ ($\Delta > 0$): dwa rozwiązania:")

st.latex(r'''
v^* = \frac{a \pm \sqrt{a^2 - 4m^2}}{2m}
''')

st.latex(r'''
v_+^* = \frac{a + \sqrt{a^2 - 4m^2}}{2m} > 1, v_-^* = \frac{a - \sqrt{a^2 - 4m^2}}{2m} < 1 
''')
st.markdown(r"Stąd dalej rozważamy $v_+^* = \frac{a + \sqrt{a^2 - 4m^2}}{2m}$.")
st.markdown(r"W dalszej analizie rozważamy zatem niezerowy, jednorodny punkt:")
st.latex(r'''
(u^*,v^*) = (\frac{2m^2}{a+\sqrt(a^2 - 4m^2)} , \frac{a + \sqrt{a^2 - 4m^2}}{2m}) 
''')



st.markdown("---")

st.markdown("### Linearyzacja i Stabilność")
st.markdown("Macierz Jacobiego $J$ dla punktu $(u^*, v^*)$:")

st.latex(r'''
J = \begin{pmatrix} f_u & f_v \\ g_u & g_v \end{pmatrix} = 
\begin{pmatrix} -1 - v^2 & -2uv \\ v^2 & 2uv - m \end{pmatrix}
''')
st.latex(r'''
det(J-\lambda I) = 0 \\
\lambda^2 - (f_u + g_v)\lambda + (f_u g_v - f_v g_u) = 0
''')
st.write(r"Na podstawie toku rozumowania przedstawionego przez Kanako Suzuki to równanie "
         "będzie miało 2 rozwiązania z ujemną częścią rzeczywistą $(Re(\lambda)<0)$ wtedy i tylko wtedy,"
         "gdy:")
st.latex(r'''
\begin{cases} 
\text{tr}(J) = f_u + g_v < 0 \\
\text{det}(J) = f_u g_v - f_v g_u > 0 
\end{cases}
''')

st.latex(r'''
tr(J) = -1 - (v^*)^2 + m < 0 \\
(v^*)^2 > m - 1
''')

st.latex(r'''
det(J) = (-1 - (v^*)^2)m + 2m (v^*)^2 > 0\\
(-1 - (v^*)^2)m + 2m (v^*)^2 > 0, m>0 \\
(-1 - (v^*)^2) + 2(v^*)^2 > 0 \\
(v^*)^2 > 1
''')
st.markdown(r"Dla naszego punktu jest zawsze spełnione, zatem stabilność $(u^*, v^*)$ mamy "
            "dla $a>2m$ oraz $(v^*)^2 > m-1$")
st.markdown("---")

st.markdown("###  Dołączenie Dyfuzji")
st.markdown(r"Niech $u(x,t) = u^* + \epsilon z(x,t)$, $v(x,t)=v^* + \epsilon w(x,t)$")
st.latex(r'''
\begin{cases} 
a - u - uv^2 = f(u,v) \\
v(uv - m) = g(u,v) 
\end{cases}
''')

st.markdown(r"Rozważmy układ równań zlinearyzowany w $(u^*,v^*)$:")
st.latex(r'''
\begin{cases} 
\frac{\partial z}{\partial t} = d_1 \frac{\partial^2 z}{\partial x^2} + \frac{\partial f}{\partial u} (u^*,v^*)z + \frac{\partial f}{\partial v} (u^*,v^*)w  \\
\frac{\partial w}{\partial t} = d_2 \frac{\partial^2 w}{\partial x^2} + \frac{\partial g}{\partial u} (u^*,v^*)z + \frac{\partial g}{\partial v} (u^*,v^*)w  \\
\frac{\partial z}{\partial t} =  \frac{\partial z}{\partial t} = 0 \\
\end{cases}
''')
st.markdown(r"Co przekłada się na macierzowe:")
st.latex(r'''
\begin{pmatrix} z \\ w \end{pmatrix} = \frac{1}{\epsilon} \begin{pmatrix} u - u^* \\ v - v^* \end{pmatrix}
''')
st.markdown(r"Układ zlinearyzowany macierzowo przyjmuje postać:")
st.latex(r'''
\frac{d}{dt} \begin{pmatrix} z \\ w \end{pmatrix} = \begin{pmatrix} d_1 & 0 \\ 0 & d_2 \end{pmatrix} \Delta
 \begin{pmatrix} z \\ w \end{pmatrix} + J(u^*, v^*) \begin{pmatrix} z \\ w \end{pmatrix}''')

st.markdown("Zakładamy, że zaburzenie występuje postaci falowej $z(x,t) = c_1 e^{\lambda t} e^{ikx}$, $w(x,t) = c_2 e^{\lambda t} e^{ikx}$.")
st.latex(r'''
\frac{d}{dt} \begin{pmatrix} z \\ w \end{pmatrix} = \begin{pmatrix} c_1  \\  c_2 \end{pmatrix} \lambda
 \begin{pmatrix} e^{\lambda t} e^{ikx} \\ e^{\lambda t} e^{ikx} \end{pmatrix} = \lambda \begin{pmatrix} z \\ w \end{pmatrix}''')
st.latex(r'''
 \begin{pmatrix} z_{xx} \\ w_{xx} \end{pmatrix} = \begin{pmatrix} c_1  \\  c_2 \end{pmatrix} (ik)^2
 \begin{pmatrix} e^{\lambda t} e^{ikx} \\ e^{\lambda t} e^{ikx} \end{pmatrix} = -k^2 \begin{pmatrix} z \\ w \end{pmatrix}''')

st.markdown("Po podstawieniu do macierzowej postaci układu zlinearyzowanego uzyskujemy:")
st.latex(r'''
\lambda \begin{pmatrix} z \\ w \end{pmatrix} = D (-k^2)
 \begin{pmatrix} z \\ w \end{pmatrix} + J \begin{pmatrix} z \\ w \end{pmatrix}''')
st.latex(r'''
\lambda \begin{pmatrix} z \\ w \end{pmatrix} = (J-k^2 D)
 \begin{pmatrix} z \\ w \end{pmatrix} ''')
st.markdown(r''' gdzie $D = \begin{pmatrix} d_1 & 0 \\ 0 & d_2 \end{pmatrix}$''')

st.markdown("Uzyskaliśmy zatem warunek na wartości własne:")

st.latex(r'''
M = J - k^2 D = \begin{pmatrix} f_u - k^2 d_1 & f_v \\ g_u & g_v - k^2 d_2 \end{pmatrix}
''')
st.markdown("Chcemy niezerowe rozwiązania układu:")
st.latex(r'''
(J - k^2 -\lambda I) = \begin{pmatrix} z \\ w \end{pmatrix} <=> \text{det}(J - k^2 D) = 0
''')

st.markdown("Uzyskujemy poniższe równanie charakterystyczne, z którego mamy relację dyspersji:")
st.latex(r'''
\lambda^2 - \text{tr}(M)\lambda + \text{det}(M) = 0 \\

\lambda(k) = \frac{tr(M) \pm \sqrt{(tr(M)^2 - 4 det(M)^2}}{2}
''')

st.markdown("Gdzie:")
st.latex(r'''
\text{tr}(M) = (f_u + g_v) - k^2(d_1 + d_2)
''')
st.latex(r'''
\text{det}(M) = d_1 d_2 k^4 - (d_1 g_v + d_2 f_u) k^2 + \text{det}(J)
''')

st.markdown(r"Warunkiem powstania wzorów (struktur) jest $\text{det}(M) < 0$ dla pewnego $k^2$ - wtedy mamy jedno rozwiązanie dodatnie, niestabilne:")
st.latex(r'''
(d_1 g_v + d_2 f_u)^2 > 4 d_1 d_2 \text{det}(J)
''')

st.markdown("Po przekształceniach zakres krytyczny liczb falowych $k$ przyjmuje teoretyczną postać:")
st.latex(r'''
k^2 \in \left[ \frac{(d_1 g_v + d_2 f_u) \pm \sqrt{(d_1 g_v + d_2 f_u)^2 - 4 d_1 d_2 \text{det}(J)}}{2 d_1 d_2} \right]
''')
st.latex(r'''
k^2 \in \left[ \frac{(d_1 m + d_2 (-1-\frac{(a+\sqrt{a^2-4m^2})^2}{4m^2}) \pm \sqrt{(d_1m + d_2 (-1-\frac{(a+\sqrt{a^2-4m^2})^2}{4m^2})^2 - 4 d_1 d_2 ((\frac{(a+\sqrt{a^2-4m^2})^2}{4m^2})m + 2m (-1-\frac{(a+\sqrt{a^2-4m^2})^2}{4m^2}) ) }}{2 d_1 d_2} \right]
''')

col1, col2 = st.columns(2)
with col1:
    st.subheader("Parametry:")
    a_val = st.slider("Opady", 0.0, 3.0, value=0.9)
    m_val = st.slider("Śmiertelność", 0.0,1.0, 0.45)
    d1_val = st.number_input("Dyfuzja wody:", value=1.0)
    d2_val = st.number_input("Dyfuzja biomasy:", value=0.01)

st.write(f"Analiza dla parametrów: $a={a_val:.2f}$, $m={m_val:.2f}$, $d_1={d1_val:.2f}$, $d_2={d2_val:.2f}$")

disp_fig = dispersion_analysis(a_val, m_val, d1_val, d2_val)
if disp_fig:
    st.pyplot(disp_fig)
    st.markdown(r"""
    Gdy krzywa $\lambda(k)$ znajduje się powyżej zera dla $k > 0$, system jest niestabilny względem zaburzeń przestrzennych. 
    Prowadzi to do powstawania struktur (pattern formation) o rozmiarze charakterystycznym dla maksimum wykresu.
    """)