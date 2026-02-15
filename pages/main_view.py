import streamlit as st


st.title('Modelowanie deterministyczne - raport z projektu')
st.write('Projekt dotyczy ekosystemów półpustynnych (drylands), w których woda jest czynnikiem krytycznie limitującym. Roślinność w takich ekosystemach formuje skomplikowane wzory przestrzenne w związku z samoorganizacją wynikającą z relacji między dostępnością wody a wzrostem biomasy. Celem projektu jest zbadanie modelu Klausmeiera-Graya-Scotta, czyli układu nieliniowych równań różniczkowych cząstkowych typu reakcja-dyfuzja. Model ten jest fundamentalnym narzędziem w ekologii teoretycznej umożliwiającym przewidywanie konsekwencji zmian klimatycznych, które mogą prowadzić do katastrofalnych zmian w ekosystemie.')
st.header("Problemy badawcze")
st.markdown("1. Ubezwymiarowienie modelu i wygenerowanie diagramu bifurkacyjnego."
            " Cel: Zbadanie zmiany stanu równowagi w zależności od parametru opadów a. \n"
            "2. Kiedy natura tworzy wzory? - Analiza niestabilności Turinga"
            " Cel: Formalna analiza zmiany stabilności stanu równowagi przez zjawisko dyfuzji \n"
            "3. Ogród wzorów i interpretacja parametrów"
            " Cel: Nadanie fizycznego sensu i morfologii rozwiązań. \n"
            "4. Wpływ wielkości środowiska na istnienie i stabilność rozwiązań")

st.header("Metodologia")
st.markdown(r"Kluczowym elementem przeprowadzenia eksperymentów jest solver KlausmeierSolver. Zastosowana w nim "
            "metoda rozwiązywania układu równań zakłada użycie schematu niejawnego do rozwiązania części dyfuzyjnej oraz jawnej do części nieliniowej."
            " Wykorzystanie schematu Eulera pozwala na wykorzystanie dużego kroku czasowego $h_t$. "
            "Ponadto zastosowano rozwiązania macierzowe takie jak macierze rzadkie i dekompozycję LU celem uzyskania możliwie jak najbardziej wydajnej implementacji "
            " i faktycznego przeprowadzenia obliczeń bardziej sprawnie niż w pętlach.")

st.header("Analiza błędu numerycznego")
st.markdown(r"Zastosowanie niejawnego schematu Eulera pozwoliło na uzyskanie zbieżności przy wybranym kroku czasowym. Schemat ten nie jest obciążony"
            r" dużym błędem numerycznym i jest bezpiecznym wyborem, by uniknąć eksplozji wyników. W symulacjach zaimplementowano sprawdzanie zbieżności "
            r" przez identyfikator zbieżności, który jest dodatkowym gwarantem stabilności numerycznej. W przypadku gdy symulacje (dla skrajnych przypadków) napotykały błąd "
            r" overflow, krok był arbitralnie zmniejszany.")
