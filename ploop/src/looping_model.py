################################################################################
## python 3.12
## Author: Léo Tarbouriech
## -*-utf8-*-

"""
This package implements:
    [1] N.B. Becker, A. Rosa, and R. Everaers, _The radial distribution function of worm-like chains_, THE EUROPEAN PHYSICAL JOURNAL E, 2010
    [2] Angelo Rosa, Nils B. Becker, and Ralf Everaers, _Looping Probabilities in Model Interphase Chromosomes_, Biophysical Journal Volume 98 June 2010 2410–2419
    [3] "Theory and simulations of condensin mediated loop extrusion in DNA, Takaky and Thirumalai, 2021"
"""

import numpy as np
from scipy.special import i0 as Bessel0
from scipy.integrate import quad
from warnings import warn
from scipy.integrate import simpson
from scipy.linalg import expm
from pathos.multiprocessing import ProcessPool as Pool
from scipy.interpolate import CubicHermiteSpline

################################## `Fondamental Constants` #####################################

c_ij = np.array(
    [[-3 / 4, 23 / 64, -7 / 64], [-1 / 2, 17 / 16, -9 / 16]], dtype=np.double
)  # exact
a = 14.054  # exact a priori (maybe from Jacobson-Stochmayer theory, after [1])
b = 0.473  # from fit (after [1])

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
        0  # WARNING: This expression is in contradiction with the expression given in the article. But if I don't do that, I cannot reproduce the figure
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


def Qi(r: np.array, kappa: np.double, EPS=1e-30) -> np.array:
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

    res = Jsyd(kappa) * A * B * C * Bessel0(D)
    res[np.abs(r) > 1] = 0

    if N > 1:
        if np.any(np.isnan(res)):
            res = np.nan_to_num(res, nan=0.0, copy=False)

    return res


def J(kappa: float, rc: float, rmin: float) -> float:
    """
    Compute the J factor.

    Inputs:
        kappa -> is the rigidity kappa = lp/L
        rc    -> is the capture radius alpha = rc/kappa
        rmin  -> is the minimal approach radius
    Output:
        J -> Like in the article of Rosa and Everaers
    """

    def func(r, kappa):
        return 3 * r**2 / rc**3 * Qi(r, kappa)

    res, abserr = quad(func, rmin, rc, args=(kappa))

    return res


def Ploop_rosa_no_force(kappa, rc, rmin) -> float:
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

    if np.abs(rc) >= 1:
        return 1
    if np.abs(rmin) >= 1:
        return 0

    def func(r, kappa):
        return 4 * np.pi * r**2 * Qi(r, kappa)

    # The Qi function is already normalised, so we don't need to divide by the norm
    res, abserr = quad(func, rmin, rc, args=(kappa))

    return res


def Ploop_rosa(
    kappa: float, alphac: float, alphamin: float, lp: float, f: float, EPS=1e-30
) -> float:
    """
    Compute the probability for the end-to-end distance to be in [rmin, rc] in the ansatz of article [1] and [2]
    Inputs:
        kappa -> is the rigidity kappa = lp/L
        alphac    -> alphac = Rc/lp
        alphamin  -> is the minimal approach radius Rmin/lp
        lp -> the persistence length
        f -> the force (experssed in [L]^-1, it is f/kbT)
        EPS -> used for regularisation
    Output:
        res -> the radial probability
    """

    rc = alphac * kappa
    rmin = alphamin * kappa
    L = lp / kappa

    if np.abs(rc) < np.abs(rmin):
        warn(
            "\nYou setted rc<rmin, this means that the radius at which LiCre can capture the DNA\n\
              strand is smaller than the exclusion radius of the chromatine.\nAre you sure that this makes sens?\n",
            RuntimeWarning,
        )
        return 0

    if np.abs(rc) >= 1.0:
        return 1

    if np.abs(rmin) >= 1.0:
        return 0

    if  L * f != 0:

        def func(r, kappa):
            return (
                4
                * np.pi
                * r**2
                * Qi(r, kappa, EPS=1e-30)
                * np.sinh(r * lp / kappa * f)
                / (r * lp / kappa * f)
            )
    else:

        def func(r, kappa):
            return 4 * np.pi * r**2 * Qi(r, kappa, EPS=1e-30)

    # The Qi function is normalised but the  Qi * exp(r * lp / kappa * f * np.cos(theta)) is not
    # So we need to compute the normalisation factor
    # and divide the result by it
    norm, _ = quad(func, 0, 1, args=(kappa))
    res, _ = quad(func, rmin, rc, args=(kappa))

    Ploop = res / (norm + EPS)
    if Ploop > 1. or Ploop < 0.:
        return np.nan
    return Ploop


def Ploop_camembert(
    kappa: np.double,
    alphac: np.double,
    rwidth: np.double,
    lp: np.double,
    f: np.double,
    thetaf: np.double,
    EPS=1.e-30,
) -> np.double:
    """
    Compute the probability for the end-to-end distance to be in [rmin, rc] in the ansatz of article [1] and [2].
    There is an additional anzatz proposed by Daniel and Nicolas which is to integrate the radial density on a camembert.
    The binding site is at one of the extremity of the camembert (BB symbolises the binding site):
                                                                 (BB                            )
                                   oooooo
                               oooo   I  oooo
                            ooo       I   .  ooo
                          oo          I  .     oo
                         oo           I .        oo
                         oo           I.         oo
                          oo          I         oo
                            ooo       I      ooo
                               oooo   I  oooo                ~~~~~
                                   ooBBoo                 ~~~
                                     BB~~~~~~~~~~~~~~~~~~~

    When alphac is very small, in the short length limit (kappa -> infinity), numerical evaluation becomes numerically unstable.
    This can lead to produce negative values for the Ploop. Then the code is made to replace negative and null values by nans.
    It is possible to get rid of these nans by interpolating them.

    Inputs:
        kappa   -> is the rigidity kappa = lp/L
        alphac  -> alphac = Rc/lp
        rwidth  -> the width (thikness) of the camembert, it is a reduced width rwidth=width/lp,
                   to have a sensible meaning it must be small compared to the persistence length.
        lp      -> the persistence length
        f       -> the force (experssed in L^-1, it is f/kbT)
        thetaf  -> the angle of the force with respect to the camembert
        EPS     -> used for regularisation

    Output:
        res     -> the radial probability
    """
    rc = alphac * kappa

    def spherosymetric_density(r, kappa):
        if f != 0:
            return (
                (4 * np.pi * r**2 * Qi(r, kappa, EPS=1e-30))
                * np.sinh(f * lp / kappa * r)
                / (f * lp / kappa * r)
            )
        else:
            return 4 * np.pi * r**2 * Qi(r, kappa, EPS=1e-30)

    def camembert_density(r, kappa):
        def func2(theta):
            x = np.sqrt(r**2 + rc**2 + 2 * r * rc * np.cos(theta))
            forceterm = np.exp(
                f * lp / kappa * r * np.cos(theta - thetaf)
                + f * rc * lp / kappa * np.cos(thetaf)
            )

            return r * Qi(x, kappa, EPS=1e-30) * forceterm

        res, _ = quad(func2, 0.0, 2.0 * np.pi)

        return res

    def I1(
        theta, kappa, f, thetaf, lp, rc
    ):  # en fait c'est I0, I2 dans mes notes manuscrites
        a = -rc * np.cos(theta) / np.sqrt(1 - rc**2 * np.sin(theta) ** 2) + 1
        b = (
            f
            * lp
            / kappa
            * (-rc * np.cos(theta) + np.sqrt(1 - rc**2 * np.sin(theta) ** 2))
            * np.cos(theta - thetaf)
        )

        return a * np.exp(b)

    def I2(
        theta, kappa, f, thetaf, lp, rc
    ):  # en fait c'est I1 dans mes notes manuscrites
        a = rc * np.cos(theta) / np.sqrt(1 - rc**2 * np.sin(theta) ** 2) + 1
        b = (
            f
            * lp
            / kappa
            * (-rc * np.cos(theta) - np.sqrt(1 - rc**2 * np.sin(theta) ** 2))
            * np.cos(theta - thetaf)
        )

        return a * np.exp(b)

    # Limit behaviour of the normalisation factor
    # for kappa -> infinity
    def spherosymetric_density_gig_kappa(kappa, f, lp):
        if f * lp / kappa == 0:
            return 4 * np.pi
        else:
            return 4 * np.pi * np.sinh(f * lp / kappa) / (f * lp / kappa)

    # Limit behaviour for kappa -> infinity
    # This threshold is arbitrary but numerically I have observed it cannot be higher that 27.
    # The higher kappa is, the slower the computation is. Because in the quadrature, it has to compute on a very dense
    # grid to follow the shape Q_i at high kappa.
    if kappa > 10.0 and rc <= 1.0:
        theta_max = np.arccos(1 / (2 * rc**2) - 1)
        res, _ = quad(I1, -theta_max, theta_max, args=(kappa, f, thetaf, lp, rc))
        res = res * np.exp(f * rc * lp / kappa * np.cos(thetaf))
        norm = spherosymetric_density_gig_kappa(kappa, f, lp)

    elif (kappa > 10.0 and rc > 1.0) or (rc > 10.0 and kappa > 1.0 and alphac < 4.) or (rc > 2. and kappa > 5. and alphac <= 4.):
        theta1 = np.arccos(1 - 1 / rc)
        theta2 = np.arccos(np.sqrt(1 - 1 / rc**2))
        theta3 = np.arccos(1 - 1 / (2 * rc**2))

        # This is always true by construction:
        assert theta1 > theta2 > theta3, "The angles are not in the right order"

        res1, _ = quad(
            I2, np.pi - theta2, np.pi + theta2, args=(kappa, f, thetaf, lp, rc)
        )
        res2, _ = quad(
            I1, np.pi - theta2, np.pi - theta3, args=(kappa, f, thetaf, lp, rc)
        )
        res3, _ = quad(
            I1, np.pi + theta3, np.pi + theta2, args=(kappa, f, thetaf, lp, rc)
        )

        # The minus sign here corresponds to an inversion of the boundary of the integral on r so the
        # integral over the dirac measure must be - 1.
        res = (-res1 + res2 + res3) * np.exp(f * rc * lp / kappa * np.cos(thetaf))
        norm = spherosymetric_density_gig_kappa(kappa, f, lp)

    else:
        res, _ = quad(camembert_density, 0.0, rc, args=(kappa))
        norm, _ = quad(spherosymetric_density, 0, 1.0, args=(kappa))

    resnorm = rwidth * kappa * res / (norm + EPS)

    if resnorm < 0.0:
        return np.nan

    return resnorm


def cleaning_ploop(ploop: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    """
    Clean the ploop array by interpolating over NaN values using a cubic Hermite spline. Cubic Hermite splines
    allow to make interpolation that respects the value of the function and its derivative at each point of evaluation of the function.

    This function is useful for ensuring that the ploop array is continuous and does not contain any NaN values.

    Remark: The same technique can be used to make the computation of Ploop faster. Use the function Ploop camembert on a
    wide gride and then interpolate the result on a finer grid. This is very usefull for instance to evaluate Ploop at a base paire level
    a each iteration of a code that make vary the force. @nicolas ;-)

    Remark: As it is writen it is mutating the input so normally you should use copy(ploop) as argument.

    Inputs:
        ploop -> the ploop array to be cleaned
        kappas -> the kappa values corresponding to the ploop array (they must be sorted in increasing order)
    Output:
        ploop -> the cleaned ploop array
    """

    finite_mask = np.isfinite(ploop)
    nan_mask = np.isnan(ploop)
    finite_ploop_kappas = kappas[finite_mask]
    finite_ploop_values = ploop[finite_mask]
    grad = np.gradient(finite_ploop_values, finite_ploop_kappas)
    interpoland = CubicHermiteSpline(
        finite_ploop_kappas, finite_ploop_values, grad, extrapolate=True
    )

    ploop[nan_mask] = interpoland(kappas[nan_mask])

    return ploop


############################################################################################
# The following functions define necessary tools  to compute the evolution of the
# probability distibution from an initial condtion to a stationnary state.
def FokkerKernel(
    Peq: np.ndarray, P0: np.ndarray, t: np.ndarray, returnKernel = False
) -> np.ndarray:
    """
    This function implement a kernel such that knowing the initial condition and the desired
    equilibrium configuration, we can compute the evolution in a effective potential.
    The effective potential is such that Peq = A.exp(-V)
    Then the kernel is:
    $$ G^{dag} = frac{d}{dx} left( Peq(x) left( frac{d}{dx} frac{•}{Peq(x)} right) right) $$

    Arguments:
     - Peq -> np.array representing the stationnary distribution you want
     - P0 -> np.array representing the intial distribution you start from
     - t -> the grid of time you want to evaluate on
     - returnKernel -> bool, whether to return the kernel that have been used. Usefull in developpement but timely in prodution.
     The point is that when doing a new situation, you want to check whether the problem is well posed and wether the kernel
     looks like what you think (symmetry,...).
    Output:
     - a matrix of size P0 * t representing P(x, t).

    NB: The evolution produced by Fokker Kernel is concervative. It means that the integral of the probability density will be concerved.
    This correspondes also to the requirement that there exists an equilibrium density Peq (the potential must be confining).
    This means that if P0 and Peq does not have the same normalisation, you will optain a limite at infinite time that does not have the
    same normalisation as Peq. This is the expected behaviour. The good way to proceed is to ensure that the integral of Peq and P0 are
    the same.
        In the case where the distribution that you use are truncated, you must thing about how to normalised.
    But still the integral over the full support on the equilibrium distribution must be the same as the integral over the
    full support of the initial distribution.
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
        if returnKernel:
            return expm(t * kernel) @ P0, kernel
        return np.einsum("kij,j->ik", expm(args), P0)

    ### Solution at time t that is a scalar:
    if returnKernel:
        return expm(t * kernel) @ P0, kernel
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
        Rc -> radius of capture
        Rmin -> minimal radius of capture
        lp -> persistence length
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

    # Interpolating
    mask = np.isnan(Peq)
    Peq_cleaned = np.copy(Peq)
    Peq_cleaned[mask] = np.interp(
        lengths[mask],
        lengths[~mask],
        Peq[~mask]
    )

    return FokkerKernel(Peq_cleaned, P0, time), Peq_cleaned


def Pjump_camembert(
    lengths: np.ndarray,
    time: float,
    P0: float,
    lp: float,
    Rc: float,
    Width: float,
    f: float,
    thetaf: float,
    EPS=1e-30,
) -> np.ndarray:
    """
    Compute the probability to jump at a given distance at a time t. The difference with Pjump is that it rely on the function Ploo_camembert instead of Ploop.

    Inputs:
        lengths -> lengths on which to compute the probability density
        time -> time at which to compute the probability density
        P0 -> initial condition
        Rc -> radius of capture
        Width -> With of the camembert
        lp -> persistence length
        f -> force (expressed as F/kbT)
        EPS -> 1e-30

    Output:
        a np.array of shape len(lengths) * len(times)
    """

    ### Computation of the equilibrium distribution:
    alphac = Rc / lp
    rwidth = Width / lp
    kappa = lp / (lengths + 1e-30)

    Peq = np.array(
        Pool().map(
            Ploop_camembert,
            kappa,
            [alphac] * len(kappa),
            [rwidth] * len(kappa),
            [lp] * len(kappa),
            [f] * len(kappa),
            [thetaf] * len(kappa),
        )
    )

    # Interpolating
    mask = np.isnan(Peq)
    Peq_cleaned = np.copy(Peq)
    Peq_cleaned[mask] = np.interp(
        lengths[mask],
        lengths[~mask],
        Peq[~mask]
    )

    return FokkerKernel(Peq_cleaned, P0, time), Peq_cleaned


################################################################################
# This part implements some function to compare the Takaki and Rosa ansatz for loop
# formation probability under force.
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
    Compute the radial density inthe rosa ansatz

    Inputs:
        r -> array of reduced radius on which to compute the probability density, must spam from
        0 to 1 for proper normalisation
        L -> lengths of the polymer, its units defines the units of the length in th simulation
        f -> force, f r L_bp = (F r L_{m}) / (k_b T ) where F is a force in N = Jm^{-1}
        kappa -> rigidity of the polymer
    Outputs:
        Probability to form a loop of length L
    """

    radial_density = (
        4 * np.pi * r**2 * Qi(r, kappa, EPS=1e-30) * np.sinh(f * r * L) / f * r * L
    )
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
