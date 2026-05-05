# Calculate Interpolation Grids
This project aims to calculate interpolation grids to accelerate the evaluation.
## Targets
### 1. Calculate velocity dispersion, given the mass profile, tracer profile and the aperture.


To calculate the velocity dispersion, we need to solve the Jeans equation. 

The mass profile is a power-law profile,whose 3-d density profile can be described as:
$$\rho(x) = \frac{3 - \gamma}{2\pi^{3/2}} \frac{\Gamma[\gamma / 2]}{\Gamma[(\gamma - 1) / 2]} \frac{M_5}{(5\mathrm{kpc})^{3- \gamma}} x^{-\gamma}$$


where $M_5$ is the total mass within a cylinder of radius 5 kpc, and $\gamma$ is the slope of the 3-d density profile. 

The tracer profile is a Sersic profile. The projected surface density profile can be described as:
$$\Sigma(R) = \Sigma_0 \exp\left[-b_n \left(\frac{R}{R_e}\right)^{1/n}\right]$$
Here we will use two types of tracer profiles.The first one is a Sersic profile, which means the index $n$ is a free parameter. The second one is a de Vaucouleurs profile, which is a special case of the Sersic profile with $n=4$.
We will predict the velocity dispersion for both two cases. 

For interpolation preparation cases, the parameters for the tracer profile is given。We want to calculate the velocity dispersion for a grid of the density slope $\gamma$. The aperture is fixed to be a rectangular aperture with a width of 1.6 arcsec and a height of 0.9 arcsec. Also, we want the calculated velocity dispersion corresponds to a specific mass normalization, which is $M_5 = 1.0 M_\odot$, such that the velocity dispersion can be easily scaled given $M_5$.

We have already had a implementation for solving the Jeans equation. The package is located in '/Users/liurongfu/tools/spherical_jeans/'. Under this directory, there are scripts for the implementation and examples for how to use the package. Another example is /Users/liurongfu/Desktop/Spectrum_reduction/make_jeans_grid.py. One should carefully read these examples to understand how to use the package.

### 2. Calculate m5 and dm5/dtheta_ein
For a power-law mass profile, given the observation of the Einstein radius, the parameter of this profile: $M_5$ and $\gamma$ is no longer independent. Instead of calculating $M_5$ every time, the goal of this section is to prepare a grid of $M_5$ given $\gamma$ and the Einstein radius observation. Also, we want to calculate the derivative of $M_5$ with respect to the Einstein radius, at given $\gamma$. 

For the specified case of a power-law mass profile, the Einstein radius can be calculated by solving the following equation:

$$\Sigma_c = \frac{c^2 D_s}{4\pi GD_l D_{ls}}$$
$$r_{\text{ein}} = \left(\frac{10^{M_5}}{\pi \Sigma_c 5^{3-\gamma}}\right)^{\frac{1}{\gamma-1}}$$

The Calculation of $D_l, D_s, D_{ls}$ can refer to /Users/liurongfu/tools/numba_friendly/cosmology.py. The preparation can refer to /Users/liurongfu/Desktop/Spectrum_reduction/make_m5_grids.py.

## Input and Output
The input file is under the directory /Users/liurongfu/Work/CMASS_lens_project/data/raw/, there two hdf5 files, which is specified by the tracer profile that should be used. Each file has several groups, which corresponds to different galaxies. For every galaxies, the interpolation grids of enclosed mass and its Einstein-radius derivative should be calculated under `mass_definitions/{m5,m10}/`, while the grid of velocity dispersion should only be calculated for galaxies whose `num_sigma` value is greater than zero. The rebuilt file keeps the same top-level galaxy grouping and the same root-level `gamma_grid`, but all mass-dependent products live only under `mass_definitions/<label>/`: `mass_grid`, `dmass_dthetaein_grid`, and, when applicable, `s2_grid`.
