### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ e486d0f7-4c49-4fc2-85da-1e149be5ae85
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 0e683651-b119-401c-9142-bb9a2a7e56c4
begin
	using Revise, RunwayLib, Plots
	using Unitful.DefaultSymbols, Rotations
	using Distributions, Unitful, Plots
end

# ╔═╡ f6f880a7-5c6b-427a-8a0e-a1adf80419ec
md""" 
# Monte Carlo Integrity Check and Calibration

We estimate the **pose integrity** of a vision-based camera localization system
by comparing *nominal* vs *empirical* uncertainty coverage levels using Monte Carlo simulation.

This notebook provides a statistical validation framework for pose-based integrity analysis:

1. **Sample** multiple noisy pose estimates.  
2. **Fit** a Gaussian distribution to the empirical data.  
3. **Compute** nominal vs empirical coverage intervals.  
4. **Plot** calibration to assess uncertainty realism.


"""

# ╔═╡ 86dad518-9b91-4dad-ac41-f1ca9409d5e5
md"""
## 1. Problem Setup

We estimate the 3D camera position

$\begin{aligned}
\mathbf{x} =
\begin{bmatrix}
x \\ y \\ z
\end{bmatrix}
\in \mathbb{R}^3
\end{aligned}$

from noisy 2D image feature measurements of known world points.

Each world point $\mathbf{p}_i^{\text{world}}$ is projected via

$\begin{aligned}
\mathbf{y}_i^{\text{true}} = \pi(\mathbf{x}_{\text{true}}, R_{\text{true}}, \mathbf{p}_i^{\text{world}})
\end{aligned}$

and observed with Gaussian noise:

$\begin{aligned}
\mathbf{y}_i^{\text{noisy}} = \mathbf{y}_i^{\text{true}} + \boldsymbol{\epsilon}_i,
\quad
\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \sigma_y^2 I_2).
\end{aligned}$

An estimator $f(\cdot)$ returns the estimated camera position

$\begin{aligned}
\hat{\mathbf{x}} = f(\{\mathbf{y}_i^{\text{noisy}}\}_{i=1}^{N_p}).
\end{aligned}$

---
"""

# ╔═╡ 6072f960-54e8-4a9b-9400-5462fd2eef00
pwd()

# ╔═╡ 594f86dc-fe32-4fa3-b305-d55edc3f7570
md"""
## 2. Sampling and Empirical Distribution

We perform \( N \) Monte Carlo runs:

$\begin{aligned}
\hat{\mathbf{x}}^{(j)} = f(\text{noisy observations}), \quad j=1,\dots,N.
\end{aligned}$

The samples are stacked into

$\begin{aligned}
X =
\begin{bmatrix}
\hat{x}^{(1)} & \hat{x}^{(2)} & \dots & \hat{x}^{(N)} \\
\hat{y}^{(1)} & \hat{y}^{(2)} & \dots & \hat{y}^{(N)} \\
\hat{z}^{(1)} & \hat{z}^{(2)} & \dots & \hat{z}^{(N)}
\end{bmatrix}.
\end{aligned}$

We then fit a multivariate Gaussian

$\begin{aligned}
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)
\end{aligned}$

using maximum likelihood:

$\begin{aligned}
\boldsymbol{\mu} = \frac{1}{N}\sum_j \hat{\mathbf{x}}^{(j)},
\quad
\Sigma = \frac{1}{N-1}\sum_j (\hat{\mathbf{x}}^{(j)} - \boldsymbol{\mu})(\hat{\mathbf{x}}^{(j)} - \boldsymbol{\mu})^T.
\end{aligned}$
"""

# ╔═╡ 95908049-0986-456e-9fdb-a8b00d54e79e
function getCamPosEst()

	runway_corners = [
	    WorldPoint(0.0m, 50m, 0m),     # near left
	    WorldPoint(3000.0m, 50m, 0m),  # far left
	    WorldPoint(3000.0m, -50m, 0m),  # far right
	    WorldPoint(0.0m, -50m, 0m),    # near right
	]
	
	cam_pos = WorldPoint(-2000.0m, 12m, 150m)
	cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)
	
	true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]
	noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]
	
	(cam_pos_est, cam_rot_est) = estimatepose6dof(
	    PointFeatures(runway_corners, noisy_observations)
	)[(:pos, :rot)]
		cam_pos_est

	return cam_pos_est
end

# ╔═╡ 29b80cf5-060d-4b80-99ea-1334f7729fbd
md"""
## 3. Coverage Calculation

For one variable of interest $v \in \{x, y, z\}$:

$\begin{aligned}
v^{(j)} \sim \mathcal{N}(\mu_v, \sigma_v^2).
\end{aligned}$

Given a nominal coverage level $c \in (0,1)$,
we compute the symmetric confidence interval:

$\begin{aligned}
[q_{\text{lo}}, q_{\text{hi}}]
= [\mu_v - z_c \sigma_v,\; \mu_v + z_c \sigma_v],
\end{aligned}$

where $z_c = \Phi^{-1}\!\left(\frac{1+c}{2}\right)$
and $\Phi^{-1}$ is the inverse standard normal CDF.

The *empirical coverage* is then:

$\begin{aligned}
\hat{c} =
\frac{1}{N}
\sum_{j=1}^{N}
\mathbf{1}(q_{\text{lo}} \le v^{(j)} \le q_{\text{hi}}),
\end{aligned}$

where \( \mathbf{1}(\cdot) \) is the indicator function.
"""

# ╔═╡ 5422fc33-bc5e-426e-a446-8824083cd240
function get_observed_coverage(num_samples=500, coverage_level=0.99, var_of_interest="x")
	
	# Define constants
	var_to_idx = Dict("x" => 1, "y" => 2, "z" => 3)

	# Get samples
	cam_pos_estimates = [getCamPosEst() for _ in 1:num_samples]
	samples = hcat([[ustrip(est.x), ustrip(est.y), ustrip(est.z)] for est in cam_pos_estimates]...)
	
	# Fit distribution
	dist_xyz = Distributions.fit_mle(MvNormal, samples)
	
	# Normalize dist in x direction
	var_idx = var_to_idx[var_of_interest]
	μ_var = mean(dist_xyz)[var_idx]
	σ_var = std(dist_xyz)[var_idx]
	dist_var = Distributions.Normal(μ_var, σ_var)
	
	# Get quantile
	q = (1 - coverage_level) / 2 + coverage_level
	var_hi = quantile(dist_var, q)
	var_lo = μ_var - (var_hi - μ_var)

	# Get observed coverage (percentage of samples within [var_lo, var_hi])
	samples_var = samples[var_idx, :]
	mask = (var_lo .<= samples_var) .& (samples_var .<= var_hi)
	num_within_samples = count(mask)
	observed_coverage = num_within_samples / num_samples

	return observed_coverage, samples_var, var_lo, var_hi
	
end

# ╔═╡ 0523beec-3f22-48f4-9ffa-3324269614e6
function create_histogram(var_of_interest = "x")

	# Get observed coverage
	observed_coverage, samples_var, var_lo, var_hi = get_observed_coverage(var_of_interest=var_of_interest)

	# Create histogram
	histogram(samples_var, bins=30, alpha=0.5, label="$var_of_interest samples")
	vline!([var_lo, var_hi], label=["$(var_of_interest)_lo" "$(var_of_interest)_hi"], color=:red, lw=2)
end

# ╔═╡ cf6e9d95-46ff-426e-a5a0-c2a98c59c438
function plot_calibration(nominal_coverage_levels, observed_coverage_levels, var_of_interest)

	# Plot calibration curve
	plot(
	    nominal_coverage_levels, observed_coverage_levels,
	    label="Empirical coverage",
	    xlabel="Nominal coverage level",
	    ylabel="Observed coverage",
	    legend=:topleft,
	    lw=2,
	    color=:blue,
	    marker=:circle,
	    title="Calibration Plot for $var_of_interest"
	)
	
	# Add 1:1 reference line (perfect calibration)
	plot!(nominal_coverage_levels, nominal_coverage_levels,
	      label="Ideal calibration",
	      color=:red,
	      linestyle=:dash,
	      lw=2)	
end

# ╔═╡ 514de880-50e8-458e-8880-293ffc780b3f
md"""
## 4. Calibration Curve

We repeat this for multiple nominal coverage levels
$c_k = 0, 0.1, 0.2, \dots, 1.0$
and obtain the corresponding empirical levels $\hat{c}_k$.

Plotting $(c_k, \hat{c}_k)$ produces the **calibration curve**.

- The red dashed line $(y = x)$ is **perfect calibration**.
- A curve **below** the line → *underconfident* (intervals too narrow).
- A curve **above** the line → *overconfident* (intervals too wide).

---
"""

# ╔═╡ 4d0d3ec7-f914-484c-95f3-caed152e0054
function check_calibration(var_of_interest="x", num_samples=500, coverage_step=0.1)

	# Define nominal coverage levels
	nominal_coverage_levels = [i for i in 0:coverage_step:1.0]

	# Calculate observed coverage levels
	observed_coverage_levels = [get_observed_coverage(num_samples, coverage_level, var_of_interest)[1] for coverage_level in nominal_coverage_levels]

	# Plot calibration
	plot_calibration(nominal_coverage_levels, observed_coverage_levels, var_of_interest)
end

# ╔═╡ 48beb3ee-e907-4407-aa78-8467f6f09443
begin
	check_calibration("x", 500, 0.01)
end

# ╔═╡ 56b8a70b-cd35-49ac-9927-567e38db6b2a
check_calibration("y", 500, 0.01)

# ╔═╡ 85a43112-ff09-439a-a0c1-ef31718616a2
check_calibration("z", 500, 0.01)

# ╔═╡ Cell order:
# ╟─f6f880a7-5c6b-427a-8a0e-a1adf80419ec
# ╟─86dad518-9b91-4dad-ac41-f1ca9409d5e5
# ╟─6072f960-54e8-4a9b-9400-5462fd2eef00
# ╟─e486d0f7-4c49-4fc2-85da-1e149be5ae85
# ╟─0e683651-b119-401c-9142-bb9a2a7e56c4
# ╟─594f86dc-fe32-4fa3-b305-d55edc3f7570
# ╠═95908049-0986-456e-9fdb-a8b00d54e79e
# ╟─29b80cf5-060d-4b80-99ea-1334f7729fbd
# ╠═5422fc33-bc5e-426e-a446-8824083cd240
# ╟─0523beec-3f22-48f4-9ffa-3324269614e6
# ╟─cf6e9d95-46ff-426e-a5a0-c2a98c59c438
# ╟─514de880-50e8-458e-8880-293ffc780b3f
# ╟─4d0d3ec7-f914-484c-95f3-caed152e0054
# ╠═48beb3ee-e907-4407-aa78-8467f6f09443
# ╠═56b8a70b-cd35-49ac-9927-567e38db6b2a
# ╠═85a43112-ff09-439a-a0c1-ef31718616a2
