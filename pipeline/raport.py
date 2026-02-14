import os

import streamlit as st
import pandas as pd
import numpy as np
st.Page("../pages/main_view.py")
pages = {
    "Wstęp": [st.Page("../pages/main_view.py", title="Opis problemu badawczego")],
    "Analiza teoretyczna": [
        st.Page("../pages/bifurcation_theory.py", title="Ubezwymiarowienie"),
        st.Page("../pages/bifurc_plt.py", title="Analiza bifurkacyjna"),
        st.Page("../pages/turing_stability.py", title="Analiza niestabilności Turinga"),
    ],
    "Symulacje": [
        st.Page("../pages/patterns_garden.py", title="Ogród wzorów"),
    ],
}

if __name__=="__main__":
    pg = st.navigation(pages)
    pg.run()