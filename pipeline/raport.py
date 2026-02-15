import streamlit as st

st.Page("../pages/main_view.py")
pages = {
    "Wstęp": [
        st.Page("../pages/main_view.py", title="Opis problemów badawczych")],
    "Analiza teoretyczna i symulacje": [
        st.Page("../pages/bifurcation_theory.py", title="Ubezwymiarowienie"),
        st.Page("../pages/bifurc_plt.py", title="Analiza bifurkacyjna"),
        st.Page("../pages/turing_stability.py", title="Analiza niestabilności Turinga"),
        st.Page("../pages/patterns_garden.py", title="Ogród wzorów"),
        st.Page("../pages/domain_size.py", title="Wpływ wielkości środowiska"),
    ],
}

if __name__=="__main__":
    pg = st.navigation(pages)
    pg.run()