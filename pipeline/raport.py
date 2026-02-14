import os

import streamlit as st
import pandas as pd
import numpy as np

st.title('Modelowanie deterministyczne - raport z projektu')
st.write('Projekt dotyczy ekosystemów półpustynnych (drylands), w których woda jest czynnikiem krytycznie limitującym. Roślinność w takich ekosystemach formuje skomplikowane wzory przestrzenne w związku z samoorganizacją wynikającą z relacji między dostępnością wody a wzrostem biomasy. Celem projektu jest zbadanie modelu Klausmeiera-Graya-Scotta, czyli układu nieliniowych równań różniczkowych cząstkowych typu reakcja-dyfuzja. Model ten jest fundamentalnym narzędziem w ekologii teoretycznej umożliwiającym przewidywanie konsekwencji zmian klimatycznych, które mogą prowadzić do katastrofalnych zmian w ekosystemie.')

pages = {
    "Analiza teoretyczna": [
        #st.Page("../notebooks/bif_clean.py", title="Ubezwymiarowienie"),
        st.Page("../notebooks/bif_clean.py", title="Analiza bifurkacyjna"),
        #st.Page("notebooks/turings_instability.py", title="Analiza niestabilności Turinga"),
    ],
    "Symulacje": [
        st.Page("../notebooks/patterns_clean.py", title="Ogród wzorów"),
    ],
}
pg = st.navigation(pages)
pg.run()