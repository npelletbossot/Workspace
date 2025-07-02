################################################################################
## Looping Probability : Theory from
##      [1] N.B. Becker, A. Rosa, and R. Everaers, _The radial distribution function of worm-like chains_, THE EUROPEAN PHYSICAL JOURNAL E, 2010
##      [2] Angelo Rosa, Nils B. Becker, and Ralf Everaers, _Looping Probabilities in Model Interphase Chromosomes_, Biophysical Journal Volume 98 June 2010 2410–2419
## python 3.12
## Author: Léo Tarbouriech
## -*-utf8-*-

## TODO: Tensorise every function

"""
This package implements:
    [1] N.B. Becker, A. Rosa, and R. Everaers, _The radial distribution function of worm-like chains_, THE EUROPEAN PHYSICAL JOURNAL E, 2010
    [2] Angelo Rosa, Nils B. Becker, and Ralf Everaers, _Looping Probabilities in Model Interphase Chromosomes_, Biophysical Journal Volume 98 June 2010 2410–2419
    [3] "Theory and simulations of condensin mediated loop extrusion in DNA, Takaky and Thirumalai, 2021"
"""

import numpy as np
from scipy.special import i0 as Bessel0
from scipy.integrate import quad as Integrate
from warnings import warn
from scipy.integrate import simpson
from scipy.linalg import expm
from pathos.multiprocessing import ProcessPool as Pool

######################### `Fondamental Constants` #########################

c_ij = np.array(
    [[-3 / 4, 23 / 64, -7 / 64], [-1 / 2, 17 / 16, -9 / 16]], dtype=np.double
)  # exact
a = 14.054  # exact a priori (maybe from Jacobson-Stochmayer theory)
b = 0.473  # from fit

######################### Simple functions that are used for the model #########################


def d(kappa):
    """
    The d factor as it is defined ini Rosa, Becker, Everaers
    All the constants are from fit, appart 1/8.
    """
    # rk: 0.125 = 1/8

    kappa = np.array(kappa)
    mask1 = kappa < 0.125
    mask2 = kappa >= 0.125
    res = np.zeros(shape=kappa.shape)
    res[mask1] = (
        0  # TODO: Check this, this expression seams in contradiction with the expression given in thge article. But if I don't do that, I cannot reproduce the figure
    )
    res[mask2] = 1 - 1 / (
        0.177 / (kappa[mask2] - 0.111) + 6.4 * (kappa[mask2] - 0.111) ** 0.783
    )  # This has no strong impact on the final result
    return res


def c(kappa, EPS=1e-18):
    """
    The c factor as difined in Rosa, Becker Everaers.
    All the constant are from fits.
    """
    # rk: 0.2 = 1/5

    return 1 - (1 + (0.38 * (kappa + EPS) ** (-0.95)) ** (-5)) ** (-0.2)


######################### Model for loopin probability #########################


def Jsyd(kappa: np.array) -> np.double:
    """
    Interpoland of the Stochmayer J-factor derived by Shimada and Yamakawa (hight stifness) and derived in the Daniel's approximation (low stifness).
    Input:
        kappa -> a float or an array of float that represents $\frac{l_p}{L}$

    Output:
        res -> has the same shape as kappa, the factor $J_syd$ evaluated at the value(s) `kappa`
    """

    kappa = np.array(kappa)
    if kappa.any() < 0:
        raise ValueError("Kappa must be positive")

    mask1 = kappa > 0.125
    mask2 = kappa <= 0.125

    res = np.zeros(shape=kappa.shape)
    res[mask1] = (
        112.04 * kappa[mask1] ** 2 * np.exp(0.246 / kappa[mask1] - a * kappa[mask1])
    )
    res[mask2] = (3 / (4 * np.pi * kappa[mask2])) ** 1.5 * (1 - 1.25 * kappa[mask2])

    return res


def Qi(r: np.array, kappa: np.double, EPS=1e-8) -> np.array:
    """
    Radial end-to-end density for the polymer model a. k. a. $Q_I(r)$ from the Rosa, Everaers, Becker.
    """
    A = ((1 - c(kappa) * r**2) / ((1 - r**2) + EPS)) ** 2.5

    r = np.array(r)
    try:
        N = len(r)
    except TypeError:
        N = 1

    r2j = np.array([r**2, r**4, r**6]).reshape((3, N))
    kappai = np.array([1 / (kappa + EPS), 1]).reshape((2, 1)).repeat(repeats=N, axis=1)
    B = np.exp(
        np.einsum(
            "ik,ijk,jk->k",
            kappai,
            c_ij.reshape((2, 3, 1)).repeat(repeats=N, axis=2),
            r2j,
        )
        / ((1 - r**2) + EPS)
    )
    C = np.exp(-(d(kappa) * kappa * a * b * (1 + b) * r**2) / ((1 - b**2 * r**2) + EPS))
    D = -(d(kappa) * kappa * a * (1 + b) * r) / ((1 - b**2 * r**2) + EPS)

    return Jsyd(kappa) * A * B * C * Bessel0(D)


def J(kappa: float, rc: float, rmin: float) -> float:
    """
    Compute the J factor.

    Inputs:
        kappa -> is the rigidity kappa = lp/L
        rc    -> is the capture radius alpha = rc/kappa
        rmin  -> is the minimal approach radius
    Output:
        J
    """

    def func(r, kappa):
        return 3 * r**2 / rc**3 * Qi(r, kappa)

    res, abserr = Integrate(func, rmin, rc, args=(kappa))

    return res


def proba_radial(kappa, rc, rmin) -> float:
    """
    Compute the probability for the end-to-end distance to be in [rmin, rc]
    Inputs:
        kappa -> is the rigidity kappa = lp/L
        rc    -> is the capture radius Rc/L, alpha = rc/kappa
        rmin  -> is the minimal approach radius Rmin/L
    Output:
        res -> the radial probability
    """

    if np.abs(rc) < np.abs(rmin):
        warn(
            "\nYou setted rc<rmin, this means that the radius at which LiCre can capture the DNA\n\
              strand is smaller than the exclusion radius of the chromatine.\nAre you sure that this makes sens?\n",
            RuntimeWarning,
        )
        return 0

    if rc >= 1:
        return 1
    if rc <= -1:
        return 1
    if rmin > 1:
        return 0
    if rmin < -1:
        return 0
    if rmin > rc:
        return 0

    def func(r, kappa):
        return 4 * np.pi * r**2 * Qi(r, kappa)

    res, abserr = Integrate(func, rmin, rc, args=(kappa))

    return res


############################################################################################
# The following functions define necessary tools  to compute the evolution of the
# probability distibution from an initial condtion to a stationnary state.
def Ploop_rosa(
    kappa: float, alphac: float, alphamin: float, lp: float, f: float, EPS=1e-30
) -> float:
    """
    Compute the probability for the end-to-end distance to be in [rmin, rc] in the ansatz of article [1] and [2]
    Inputs:
        kappa -> is the rigidity kappa = lp/L
        alphac    -> alphac = Rc/lp
        alphamin  -> is the minimal approach radius Rmin/lp
        f -> the force (experssed in L^-1, it is f/kbT)
        lp -> the persistence length
        EPS -> used for regularisation
    Output:
        res -> the radial probability
    """

    rc = alphac * kappa
    rmin = alphamin * kappa

    if np.abs(rc) < np.abs(rmin):
        warn(
            "\nYou setted rc<rmin, this means that the radius at which LiCre can capture the DNA\n\
              strand is smaller than the exclusion radius of the chromatine.\nAre you sure that this makes sens?\n",
            RuntimeWarning,
        )
        return 0

    if rc >= 1:
        return 1
    if rc <= -1:
        return 1
    if rmin > 1:
        return 0
    if rmin < -1:
        return 0
    if rmin > rc:
        return 0

    def func(r, kappa):
        return 10 * np.pi * r**2 * Qi(r, kappa, EPS=1e-30) * np.exp(r * lp / kappa * f)

    norm, abserr = Integrate(func, 0, 1, args=(kappa))
    res, abserr = Integrate(func, rmin, rc, args=(kappa))
    return res / (norm + EPS)


def FokkerKernel(Peq: np.ndarray, P0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    This function implement a kernel such that knowing the initial condition and the desired
    equilibrium configuration, we can compute the evolution in a effective potential.
    The effectivee potential is such that Peq = A.exp(-V)
    Then the kernel is:
    $$ G^{\dag} = \frac{d}{dx} \left( Peq(x) \left( \frac{d}{dx} \frac{•}{Peq(x)} \right) \right) $$
    """

    ### Computation of the kernel:
    gradx = np.zeros(shape=(len(Peq), len(Peq)))
    for i in range(gradx.shape[0]):
        gradx[i, i] = -1
        gradx[i, (i + 1) % gradx.shape[0]] = 1
    gradx[-1, 0] = 0
    gradx[0, -1] = 0
    gradx[-1, :] = gradx[-2, :]

    kernel = np.einsum("ik,k,kj,j->ij", -gradx.T, Peq, gradx, 1 / (Peq))

    if type(t) is np.ndarray:
        args = np.einsum("ij,k->kij", kernel, t)
        return np.einsum("kij,j->ik", expm(args), P0) # Only returning what we need

    ### Solution at time t:
    return expm(t * kernel) @ P0


def Pjump(
    lengths: np.ndarray,
    time: float,
    P0: float,
    lp: float,
    Rc: float,
    Rmin: float,
    f: float,
    EPS=1e-30,
) -> np.ndarray:
    """
    Compute the probability to jump at a given distance at a time t.

    Inputs:
        lengths -> lengths on which to compute the probability density
        time -> time at which to compute the probability density
        P0 -> initial condition
        lp -> persistence length
        Rc -> radius of capture
        Rmin -> minimal radius of capture
        f -> force (expressed as F/kbT)
        EPS -> 1e-30

    Output:
        a np.array of shape len(lengths) * len(times)
    """

    ### Computation of the equilibrium distribution:
    alphac = Rc / lp
    alphamin = Rmin / lp
    kappa = lp / (lengths + 1e-30)

    Peq = np.array(
        Pool().map(
            Ploop_rosa,
            kappa,
            [alphac] * len(kappa),
            [alphamin] * len(kappa),
            [lp] * len(kappa),
            [f] * len(kappa),
        )
    )

    return FokkerKernel(Peq, P0, time)


def takaki_ansatz(r: np.array, L: float, f: float, kappa: float, EPS=1e-30):
    """
    This returns exactlly the equation (2) in ref [3]

    Inputs:
        r -> array of reduced radius on which to compute the probability density
        L -> lengths of the polymer, its units defines the units of the length in th simulation
        f -> force, f r L_bp = (F r L_{m}) / (k_b T ) where F is a force in N = Jm^{-1}
        kappa -> rigidity of the polymer
    Outputs:
        Probability to form a loop of length L
    """
    t = 1.5 / (kappa + EPS)
    alpha = 0.75 * t
    N2 = (
        4
        * alpha ** (1.5)
        * np.exp(alpha)
        / (np.pi ** (1.5) * (4 + 12 * alpha ** (-1) + 15 * alpha ** (-2)))
    ) ** 2
    term1 = N2 * r * r / L * ((1 - r**2) + EPS) ** (-9 / 2)
    term2 = np.exp(-3 * t / (4 * (1 - r**2) + EPS))
    term3 = np.exp(f * r * L)

    PL = term1 * term2 * term3
    PL = PL / simpson(PL, x=r)
    return PL


def takaki_extension(r: np.array, L: float, f: float, kappa: float):
    """
    Compute the force extension relation based on the Takaki ansatz.

    Inputs:
        r -> reduced radius to sample
        L -> length of the polymer, the units of this number defines the units
        of length in the simulation
        f -> force, f r L_bp = (F r L_{m}) / (k_b T ) where F is a force in N = Jm^{-1},
        kbT in J and L_m in meters
        kappa -> rigidity of the polymer

    Outputs:
        extension of the polymer along the direction of the force (Avg(R))
    """
    Pr = takaki_ansatz(r, L, f, kappa)
    extension = simpson(r * Pr, x=r)
    return extension


def rosa_ansatz(r: np.array, L: float, f: float, kappa: float):
    """
    Compute the probability the radial density inthe rosa ansatz

    Inputs:
        r -> array of reduced radius on which to compute the probability density, must spam from
        0 to 1 for proper normalisation
        L -> lengths of the polymer, its units defines the units of the length in th simulation
        f -> force, f r L_bp = (F r L_{m}) / (k_b T ) where F is a force in N = Jm^{-1}
        kappa -> rigidity of the polymer
    Outputs:
        Probability to form a loop of length L
    """

    radial_density = 4 * np.pi * r**2 * Qi(r, kappa, EPS=1e-30) * np.exp(f * r * L)
    radial_density = radial_density / simpson(radial_density, x=r)
    return radial_density


def rosa_extension(r: np.array, L: float, f: float, kappa: float):
    """
    Compute the extension of the polymer (end-to-end distance) projected on
    the direction of the force f.

    Inputs:
        r -> reduced radius to sample
        L -> length of the polymer, the units of this number defines the units
        of length in the simulation
        f -> force, f r L_bp = (F r L_{m}) / (k_b T ) where F is a force in N = Jm^{-1},
        kbT in J and L_m in meters
        kappa -> rigidity of the polymer
    Outputs:
        extension
    """

    Pr = rosa_ansatz(r, L, f, kappa)
    extension = simpson(r * Pr, x=r)
    return extension
