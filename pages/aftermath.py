import streamlit as st


st.title('Podsumowanie')

st.header("Co oznacza histereza dla ochrony przyrody?")
st.markdown(r"Histereza, czyli zjawisko zależności aktualnego stanu układu od stanów w poprzedzających "
            r" chwilach, a także opóźnienie w reakcji na czynnik zewnętrzny, ma znaczącą rolę w ochronie przyrody. "
            r" W przeprowadzonych symulacjach zauważyliśmy, że w kontekście eksperymentu ekologicznego efekt histerezy "
            r"uwidacznia się, gdy układ przechodzi między dwoma stanami stabilnymi systemu. Dokładnie to zjawisko pozwala "
            r" odrodzić się - w rozważanym kontekście - ze stanu upustynnienia. Przy tym w związku z konkurencją w ramach biomasy "
            r" obserwowane jest tworzenie wzorów, wynikających z samoorganizacji roślinności walczącej o dostęp do wody.")

st.header("Czy system może się zregenerować po przejściowej suszy?")
st.markdown(r" Zjawisko, które obserwujemy, pozwala twierdzić, że system może się zregenerować po przejściowej suszy, "
            r" ale jest bardzo zależne od wielkości opadów obserwowanych w danym obszarze oraz jego wielkości. Większe obszary "
            r" i większe wielkości opadów pozwalają łatwiej osiągnąć odrodzenie roślinności, które może nie być możliwe dla zbyt "
            r" małej powierzchni terenu ochrony przyrody.")

st.header("Natura tworzy wzory")
st.markdown(r""" Przeprowadzone eksperymenty pozwalają zauważyć, że różne wartości parametrów pozwalają systemowi układać się w regularne
             wzory. Przy niewielkim poziomie opadów i dużej dyfuzji wody można uzyskać niewielkie kępki, które formują się w cętki. 
             Duże opady i relatywnie mniejsza dyfuzja wody to przyczynek do formowania się labiryntów, gdzie ruchomość biomasy dyktuje stosunkowo 
             większa dyfuzja biomasy. Na pośrednim poziomie opadów, przy najmniejszym poziomie dyfuzji, biomasa "rozlewa" się na całą przestrzeń poza 
              pojedynczymi punktami, tworząc dziury. Jest to wynik utraty stabilności stanów stacjonarnych, co zostało dogłębnie przedstawione 
              w teoretycznej analizie niestabilności Turinga. """)

st.subheader("Źródła")
st.markdown(r"""W tworzeniu pracy wspomagałam się sztuczną inteligencją, w szczególności przy debugowaniu, optymalizacji bądź ulepszeniu kodu oraz nauce 
 możliwości Streamlit jako dodatek do dokumentacji i dostępnych tutoriali. Ponadto korzystałam z poniżej wymienionych źródeł. """)
st.page_link("https://www.researchgate.net/publication/373183341_On_a_generalized_Klausmeier_model", label="On a generalized Klausmeier model - Martha Paola Cruz"
                                                                                                           " de la Cruz, Daniel Alfonso Santiesteban, Luis Miguel Martin Alvarez, Ricardo "
                                                                                                           "Abreu Blaya, Hernandez-Gomez Juan Carlos")
st.page_link("https://www.researchgate.net/publication/266195559_Mechanism_Generating_Spatial_Patterns_in_Reaction-Diffusion_Systems", label="Mechanism Generating Spatial Patterns in Reaction-Diffusion Systems - Kanako Suzuki")
st.page_link("https://www.sciencedirect.com/science/article/pii/S0022247X20310234?via%3Dihub", label="Bifurcation and pattern formation in diffusive Klausmeier-Gray-Scott model of water-plant interaction - Xiaoli Wang, Junping Shi, Guohong Zhang")
st.page_link("https://www.science.org/doi/10.1126/science.284.5421.1826", label="Regular and Irregular Patterns in Semiarid Vegetation - Christopher A. Klausmeier")



