import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.functions import pattern_plotting
import matplotlib
matplotlib.use('Agg')
import streamlit as st


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

fig = None

with col1:
    st.subheader("Parametry:")
    a_val = st.slider("Opady", 0.0, 3.0, value=0.9)
    m_val = st.slider("Śmiertelność", 0.0,1.0, 0.45)
    d1_val = st.number_input("Dyfuzja wody:", value=1.0)
    d2_val = st.number_input("Dyfuzja biomasy:", value=0.01)
    n = st.number_input("Podział obszaru", min_value=10, max_value=500, step=1)
    l = st.number_input("Wielkość obszaru", min_value=10, max_value=500, step=1)
    ht = st.number_input("Krok czasowy:", value=0.005, min_value=0.000, max_value=1.000, step=0.001, format="%.3f")
    if st.button("Sprawdź wzór dla parametrów"):
        fig, variance = pattern_plotting(a_val, m_val, d1_val, d2_val, n, n, l, l, ht)
with col2:
    st.subheader("Gotowe wzory:")
    if st.button("Labirynty"):
        fig, variance  = pattern_plotting(2.0,0.45,182.5,0.25, 100, 100, 100, 100, 0.1)
    if st.button("Cętki"):
        fig, variance  = pattern_plotting(1.2,0.45,80.0,0.1, 100, 100, 50, 50, 0.005)
    if st.button("Dziury"):
        fig, variance  = pattern_plotting(1.85, 0.45, 40.0, 0.01, 100,100,500,500, 0.005)
if fig is not None:
    st.pyplot(fig)

st.header("Ilościowa ocena różnorodności patternów")
st.markdown("Ocenę ilościową różnorodności patternów wykonamy za pomocą wariancji przestrzennej biomasy.")
st.info("Poniżej wygenerują się trzy wykresy, które będą podstawą analizy ilościowej i oceny różnorodności patternów. Czas wykonywania symulacji to około 6 minut.")
fig_labirynt, variance_labirynt = pattern_plotting(2.0,0.45,182.5,0.25, 100, 100, 100, 100, 0.1)
fig_dziury, variance_dziury = pattern_plotting(1.9, 0.45, 45.0, 0.01, 100,100,500,500, 0.005)
fig_cetki, variance_cetki = pattern_plotting(1.2,0.45,80.0,0.1, 100, 100, 100, 100, 0.005)

c1, c2, c3 = st.columns(3)
with c1:
    st.pyplot(fig_cetki)
with c2:
    st.pyplot(fig_labirynt)
with c3:
    st.pyplot(fig_dziury)

st.markdown(f"""Wariancja przestrzenna biomasy $Var(v)$ dla wzorców: \n
            cętki: {variance_cetki:.4f} \n
            labirynt: {variance_labirynt:.4f} \n 
            dziury: {variance_dziury:.4f}""")

st.markdown("Zauważmy, że największą wariancją charakteryzują się cętki - wzory te formują się, gdy opadów jest najmniej, ale dyfuzja wody relatywnie duża."
            " Jest to zgodne z intuicją, gdy mamy dużo kępek pośród pustyni, różnice w występowaniu biomasy na stosunkowo małej przestrzeni potrafią być"
            " bardzo duże. Z pewnością wpływa na to również punkt startowy bliski stanu stacjonarnego, którego wybór skutkował dużymi wartościami gęstości biomasy. \n "
            "W przypadku labiryntów mamy znacząco mniejszą wariancję. Tworzą się one przy największych opadach spośród otrzymanych wzorów i bardzo dużej dyfuzji wody."
            " To pozwoliło na przemieszczanie się wody w taki sposób, że ukształtowały się regularne wzory przypominające labirynty. Wpływ na to miała również stosunkowo duża dyfuzja roślinności. \n"
            "Najmniejszą wariancję wykazały dziury - powstają przy dużych opadach i przy stosunkowo dużej dyfuzji wody do dyfuzji biomasy. To pozwala na rozprzestrzenianie się "
            "roślinności, która pozostawia pustynne obszary w znaczącej mniejszości obszaru, w zasadzie punktowo. ")


