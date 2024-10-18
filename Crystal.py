import numpy as np
from sympy import Symbol, diff
import sympy
import math

class Crystal:
    """
    Represents the properties of the nonlinear crystal used in the simulation.
    """
    def __init__(self, Lc, Lo, T, w, mstart):
        self.Lc = Lc        # Coherence length (m)
        self.Lo = Lo        # Physical length of the crystal (m)
        self.T = T          # Temperature (°C)
        self.w = w          # Domain width parameter (m)
        self.mstart = mstart  # Starting index for the apodization algorithm

        # Constants
        self.nm = 1e-9
        self.um = 1e-6
        self.mm = 1e-3

        # Temperature Expansion of crystal
        # (S. Emanueli & A. Arie, App. Opt, vol. 42, No. 33 (2003))
        self.L = self.Lo * (1 + 6.7e-6 * (self.T - 25) + 11e-9 * (self.T - 25) ** 2)

        # Poling pattern attributes (to be computed)
        self.sarray = None
        self.atarray = None
        self.amuparray = None
        self.amdownarray = None
        self.altered_z = None
        self.z = None

    def refractive_index_y(self, x):
        T = self.T
        ny2 = 2.09930 + (0.922683 / (1 - (0.0467695 / x ** 2))) - (0.0138408 * x ** 2)
        Sny = np.sqrt(ny2)
        n1_y = 6.28977e-6 + 6.3061e-6 / x - 6.0629e-6 / x ** 2 + 2.6486e-6 / x ** 3
        n2_y = -0.14445e-8 + 2.2244e-8 / x - 3.5770e-8 / x ** 2 + 1.3470e-8 / x ** 3
        deln_Y = n1_y * (T - 25) + n2_y * (T - 25) ** 2
        y = Sny + deln_Y
        return y

    def refractive_index_z(self, x):
        T = self.T
        nz2 = (
            2.12725
            + 1.18431 / (1 - 5.14852e-2 / x ** 2)
            + 0.6603 / (1 - 100.00507 / x ** 2)
            - 9.68956e-3 * x ** 2
        )
        Snz = np.sqrt(nz2)
        n1_z = 9.9587e-6 + 9.9228e-6 / x - 8.9603e-6 / x ** 2 + 4.1010e-6 / x ** 3
        n2_z = -1.1882e-8 + 1.0459e-7 / x - 9.8136e-8 / x ** 2 + 3.1481e-8 / x ** 3
        deln_Z = n1_z * (T - 25) + n2_z * (T - 25) ** 2
        y = Snz + deln_Z
        return y

    def group_index_y(self, x1):
        x = Symbol("x")
        T = self.T
        ny2 = 2.09930 + (0.922683 / (1 - (0.0467695 / x ** 2))) - (0.0138408 * x ** 2)
        Sny = sympy.sqrt(ny2)
        n1_y = 6.28977e-6 + 6.3061e-6 / x - 6.0629e-6 / x ** 2 + 2.6486e-6 / x ** 3
        n2_y = -0.14445e-8 + 2.2244e-8 / x - 3.5770e-8 / x ** 2 + 1.3470e-8 / x ** 3
        deln_Y = n1_y * (T - 25) + n2_y * (T - 25) ** 2
        ny = Sny + deln_Y
        y1 = ny - x * diff(ny, x)
        y = y1.subs({x: x1})
        return float(y)

    def group_index_z(self, x1):
        x = Symbol("x")
        T = self.T
        nz2 = (
            2.12725
            + 1.18431 / (1 - 5.14852e-2 / x ** 2)
            + 0.6603 / (1 - 100.00507 / x ** 2)
            - 9.68956e-3 * x ** 2
        )
        Snz = sympy.sqrt(nz2)
        n1_z = 9.9587e-6 + 9.9228e-6 / x - 8.9603e-6 / x ** 2 + 4.1010e-6 / x ** 3
        n2_z = -1.1882e-8 + 1.0459e-7 / x - 9.8136e-8 / x ** 2 + 3.1481e-8 / x ** 3
        deln_Z = n1_z * (T - 25) + n2_z * (T - 25) ** 2
        nz = Snz + deln_Z
        z1 = nz - x * diff(nz, x)
        y = z1.subs({x: x1})
        return float(y)

    def gtarget(self, z, L, lc):
        return np.exp(-((z - L / 2) ** 2) / (L ** 2 / 8))

    def Atarget(self, w, m, L, Lc, DeltaK):
        z = np.linspace(0, m * w, num=m)
        y = (
            self.gtarget(z, L, Lc / 2)
            * np.cos(np.pi / (Lc / 2) * z)
            * np.exp(1j * DeltaK * z)
        )
        return -1j * np.trapz(y, z)

    def Am2(self, w, altered_z, m, Lc, sn):
        if len(sn) == m:
            y = np.sum(sn * np.exp(1j * np.pi / (Lc / 2) * altered_z))
            return Lc / (2 * np.pi) * (np.exp(-1j * np.pi / (Lc / 2) * w) - 1) * y
        else:
            raise ValueError("Poling array length wrong.")

    def generate_poling(self,laser,mode='periodic',resolution=5):
        if mode == 'periodic':
            self.generate_periodic_poling(resolution=resolution)
        elif mode == 'custom':
            self.generate_custom_poling(laser)

    def generate_periodic_poling(self, resolution=5):
        Lc = self.Lc
        Lo = self.Lo
        num_domains = int(np.floor(Lo / Lc))
        # Create the polarizations array using np.tile
        polarizations = np.tile([1, -1], num_domains)
        # Create the sarray using np.repeat
        self.sarray = np.repeat(polarizations, resolution)
        # Adjust Lo to be an integer multiple of Lc
        self.Lo = num_domains * Lc
        # Calculate z values directly based on the length of sarray
        self.z = np.linspace(-self.L / 2, self.L / 2, len(self.sarray))

    def generate_custom_poling(self, laser):
        # Compute refractive indices at central wavelengths
        lambda_w = laser.lambda_w
        lambda_2w = laser.lambda_2w

        nyD = self.refractive_index_y(lambda_w * 1e6)
        nzD = self.refractive_index_z(lambda_w * 1e6)
        nyP = self.refractive_index_y(lambda_2w * 1e6)

        # Compute group indices
        NyP = self.group_index_y(lambda_2w * 1e6)
        NzD = self.group_index_z(lambda_w * 1e6)
        NyD = self.group_index_y(lambda_w * 1e6)
        self.Nmean = (NzD + NyD) / 2.0

        # Compute DeltaK_0
        self.DeltaK_0 = 2 * np.pi * (nyP / lambda_2w - nzD / lambda_w - nyD / lambda_w)

        # Proceed with the apodization algorithm using self.DeltaK_0
        w = self.w
        mstart = self.mstart
        L = self.L
        Lc = self.Lc
        DeltaK = self.DeltaK_0

        num_iterations = math.ceil(L / w) + 1  # Total number of iterations

        # Precompute altered_z
        altered_z = np.linspace(0, num_iterations * w, num_iterations + 1)
        # Initialize sarray
        sarray = np.zeros(num_iterations + 1, dtype=int)
        sarray[0] = 1
        atarray = np.zeros(num_iterations, dtype=complex)
        amuparray = np.zeros(num_iterations, dtype=complex)
        amdownarray = np.zeros(num_iterations, dtype=complex)

        for idx in range(num_iterations):
            m = mstart + idx

            # Compute Atarget once per iteration
            at = self.Atarget(w, m, L, Lc, DeltaK)

            # Test with sarray[idx + 1] = 1 (up)
            sarray[idx + 1] = 1
            amup = self.Am2(w, altered_z[: idx + 2], m, Lc, sarray[: idx + 2])

            # Test with sarray[idx + 1] = -1 (down)
            sarray[idx + 1] = -1
            amdown = self.Am2(w, altered_z[: idx + 2], m, Lc, sarray[: idx + 2])

            # Compute errors
            eup = np.abs(at - amup)
            edown = np.abs(at - amdown)

            # Store results
            atarray[idx] = at
            amuparray[idx] = amup
            amdownarray[idx] = amdown

            # Decide which orientation minimizes the error
            if eup < edown:
                sarray[idx + 1] = 1  # Keep 'up' orientation
            else:
                sarray[idx + 1] = -1  # Keep 'down' orientation

        self.sarray = sarray
        self.atarray = atarray
        self.amuparray = amuparray
        self.amdownarray = amdownarray
        self.altered_z = altered_z
        self.z = altered_z - L / 2