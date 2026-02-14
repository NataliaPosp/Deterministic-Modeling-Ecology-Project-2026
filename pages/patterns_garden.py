import sys
import os

# Dodaje folder główny projektu do ścieżek wyszukiwania
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.solver import KlausmeierSolver
from pipeline.functions import pattern_plotting

import matplotlib
matplotlib.use('Agg')
import streamlit as st

###

st.title("Ogród wzorów")
st.markdown("""
### Interpretacja parametrów""")
st.markdown(r"W wersji wymiarowej modelu Klausmeiera występuje 5 parametrów, które mają swoją"
            " interpretację biologiczną. \n"
            " * **A** - wielkość **opadów** \n "
            " * **L** - wielkość **parowania wody** \n "
            " * **M** - **śmiertelność** biomasy \n "
            " * **J** - **plon** biomasy roślinnej na jednostkę pobranej wody \n"
            " * **R** - współczynnik **poboru wody** przez biomasę")

st.header("Ogród wzorów")
st.markdown("W zależności od wyboru parametrów możemy uzyskać różne wzory. Gdy mamy niestabilność"
            " Turinga, biomasa układa się w charakterystyczny sposób, co rozważaliśmy w poprzednich analizach teoretycznych.")

col1, col2 = st.columns(2)


with col1:
    st.subheader("Parametry:")
    a_val = st.slider("Opady", 0.0, 3.0, value=0.9)
    m_val = st.slider("Śmiertelność", 0.0,1.0, 0.45)
    d1_val = st.number_input("Dyfuzja wody:", value=1.0)
    d2_val = st.number_input("Dyfuzja biomasy:", value=0.01)
    if st.button("Sprawdź wzór dla parametrów"):
        fig = pattern_plotting(a_val, m_val, d1_val, d2_val, 100, 100, 100, 100, 0.1)
with col2:
    st.subheader("Gotowe:")
    if st.button("Labirynty"):
        fig = pattern_plotting(2.0,0.45,182.5,0.25, 100, 100, 100, 100, 0.1)
    if st.button("Cętki"):
        fig = pattern_plotting(1.2,0.45,80.0,0.1, 100, 100, 50, 50, 0.005)

if fig is not None:
    st.pyplot(fig)




