import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu


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

        # Konwersja na LU celem przyspieszenia
        self.solve_u = splu(A_u.tocsc())
        self.solve_v = splu(A_v.tocsc())

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

        u_next = self.solve_u.solve(right_u)
        v_next = self.solve_v.solve(right_v)
        u_next[self.bound] = 0
        v_next[self.bound] = 0

        return u_next, v_next

    def initial(self, initial_state='constant'):
        """
        Zdefiniowanie warunków początkowych do rozwiązania układu równań
        :param initial_state: 'constant' - rozpoczęcie z wysokiego poziomu biomasy
                            'perturbed' - zaburzenie w środku siatki celem odwróconego procesu rozwiązania
        :return: warunek początkowy na u, v
        """
        u_start = 2.0 * np.ones(self.Nx * self.Ny)

        if initial_state == 'perturbed':
            v_start = np.ones(self.Nx*self.Ny)
            v_view = v_start.reshape((self.Nx, self.Ny))
            v_view[self.Nx // 4:3 * self.Nx // 4, self.Ny // 4:3 * self.Ny // 4] = 2.0
            v_start = v_view.flatten()
        else:
            v_start = 2.0 * np.ones(self.Nx * self.Ny)

        return u_start, v_start

    def steps_to_steady_state(self, u, v, a, m, A_u, A_v, tolerance=0.0001, iter=1000):
        diff = 1.0
        iter_count = 0

        # Poszukiwanie zbieżności do stanu stacjonarnego
        while diff > tolerance and iter_count < iter:
            iter_count += 1
            # Krok
            u_next, v_next = self.solve_step(u, v, a, m, A_u, A_v)

            diff = np.max(np.abs(v_next - v))
            u, v = u_next, v_next

        return u, v

    def solution_for_bifurcation(self, a_vals, m, d1, d2, initial_state='constant'):
        """
        Przeprowadzenie całego numerycznego wyliczenia rozwiązań w eksperymencie badania bifurkacji.
        """

        A_u, A_v = self.evolution_matrix(d1, d2)

        v_max_down = []
        v_mean_down = []

        u, v = self.initial(initial_state)

        non_zero_u = None
        non_zero_v = None
        min_a = None

        print("Simulation for a descending...")
        for a in tqdm.tqdm(a_vals):
            u, v = self.steps_to_steady_state(u, v, a, m, A_u, A_v)

            if np.max(v) > 1.0:
                non_zero_u = u.copy()
                non_zero_v = v.copy()
                min_a = a

            v_max_down.append(np.max(v))
            v_mean_down.append(np.mean(v))

        v_max_up = []
        v_mean_up = []
        a_vals_incr = a_vals[::-1]

        #if non_zero_v is not None:
        #    u, v = non_zero_u.copy(), non_zero_v.copy()
        #else:
        #   u, v = self.initial("perturbed")

        print("Simulation for a increasing...")

        u = 2.0 * np.ones(self.Nx * self.Ny)
        v = 2.0 * np.ones(self.Nx * self.Ny)
        start_a = 0.9

        for a in tqdm.tqdm(a_vals_incr):
            if a < start_a:
                v_max_up.append(0.0)
                v_mean_up.append(0.0)
                continue

            u, v = self.steps_to_steady_state(u, v, a, m, A_u, A_v, iter=5000)
            v_max_up.append(np.max(v))
            v_mean_up.append(np.mean(v))

        return v_max_down, v_mean_down, v_max_up, v_mean_up

    def tip_point(self, a_vals, v_down, v_up):
        a_vals_down = a_vals
        a_vals_up = a_vals[::-1]

        tip_point_down = None
        for i in range(len(v_down)):
            if v_down[i] < 0.1:
                tip_point_down = a_vals_down[i]
                break

        return tip_point_down