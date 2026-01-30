import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sparse
from scipy.sparse.linalg import spsolve

class KlausmeierSolver:
    """
    Klasa rozwiązująca układ równań reakcji-dyfuzji na podstawie modelu Klausmeiera
    na siatce 2D
    """

    def __init__(self, Nx=41, Ny=41, Lx=10, Ly=10, ht=0.005):
        # Siatka
        self.Nx, self.Ny = Nx, Ny
        self.ht = ht

        # Przestrzeń
        self.x = np.linspace(0,Lx,Nx)
        self.y = np.linspace(0,Ly,Nx)
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.X_flat = self.X.flatten()
        self.Y_flat = self.Y.flatten()
        self.hx = self.x[1] - self.x[0]
        self.hy = self.y[1] - self.y[0]

        # Boundary conditions (Dirichlet)

        self.bound = (self.X_flat == Lx) | (self.X_flat == 0) | (self.Y_flat == Ly) | (self.Y_flat == 0)
        self.L = self.laplacian()

    def laplacian(self):
        """
        Metoda budująca Laplasjan wraz z pomocniczą metodą budującą macierz różniczkowania (rzadką)
        :return: Laplasjan
        """
        def D2_sparse(N):
            main_diag = -2 * np.ones(N)
            off_diag = np.ones(N - 1)
            return sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))
        id_x = sparse.eye(self.Nx)
        id_y = sparse.eye(self.Ny)
        D2_x = D2_sparse(self.Nx)
        D2_y = D2_sparse(self.Ny)

        return (sparse.kron(id_y, D2_x)/self.hx**2 + sparse.kron(D2_y, id_x)/self.hy**2).tocsr()

    def evolution_matrix(self, d1, d2):
        """
        Funkcja generująca macierz ewolucji dla ustalonych współczynników dyfuzji.
        :param d1: parametr dyfuzji dla biomasy
        :param d2: parametr dyfuzji dla wody (opadów)
        :return: macierze ewolucji A_u, A_v, odpowiednio dla biomasy i wody
        """
        id = sparse.eye(self.Nx * self.Ny)
        A_u = (id - self.ht*d1*self.L).tolil()
        A_v = (id - self.ht*d2*self.L).tolil()

        # Boundary Conditions (Dirichlet)

        for i in np.where(self.bound)[0]:
            A_u[i, :] = 0
            A_u[i, i] = 1
            A_v[i, :] = 0
            A_v[i, i] = 1

        return A_u.tocsr(), A_v.tocsr()

    def solve_step(self, u, v, a, m, A_u, A_v):
        """
        Pojedynczy krok iteracji szukającej rozwiązań
        :param u: ubezwymiarowiona woda
        :param v: ubezwymiarowiona biomasa
        :param a: parametr opadów
        :param m: mortality (śmiertelność roślin)
        :param A_u: macierz ewolucji dla u
        :param A_v: macierz ewolucji dla v
        :return: wynik jednej iteracji rozwiązania układu równań
        """

        f_u = a - u - u * ((v)**2)
        f_v = u*(v)**2 - m*(v)

        right_u = u + self.ht * f_u
        right_v = v + self.ht * f_v

        right_u[self.bound] = 0
        right_v[self.bound] = 0

        u_next = spsolve(A_u, right_u)
        v_next = spsolve(A_v, right_v)

        return u_next, v_next

    def initial(self, initial_state='constant'):
        """
        Zdefiniowanie warunków początkowych do rozwiązania układu równań
        :param initial_state: 'constant' - rozpoczęcie z wysokiego poziomu biomasy
                            'perturbed' - zaburzenie w środku siatki celem odwróconego procesu rozwiązania
        :return: warunek początkowy na u, v
        """
        u_start = 2.0 * np.ones(self.Nx * self.Ny)
        v_start = np.ones(self.Nx * self.Ny)

        if initial_state == 'perturbed':
            v_view = v_start.reshape((self.Nx, self.Ny))
            v_view[self.Nx // 4:3 * self.Nx // 4, self.Ny // 4:3 * self.Ny // 4] = 2.0

        return u_start, v_start


    def solution_for_bifurcation(self, a_vals, m, d1, d2, initial_state='constant'):
        """
        Przeprowadzenie całego numerycznego wyliczenia rozwiązań w eksperymencie badania bifurkacji.
        """
        A_u, A_v = self.evolution_matrix(d1, d2)
        v_max_list = []
        u, v = self.initial(initial_state)

        for a in tqdm.tqdm(a_vals):
            diff = 1.0
            iter_count = 0

            while diff > 0.0001 and iter_count < 1000:
                iter_count += 1
                u_next, v_next = self.solve_step(u,v,a,m,A_u,A_v)

                diff = np.max(np.abs(v_next - v))
                u, v = u_next, v_next

            v_max_list.append(np.max(v))
        return v_max_list
