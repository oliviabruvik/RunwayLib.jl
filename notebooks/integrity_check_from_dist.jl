### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ e486d0f7-4c49-4fc2-85da-1e149be5ae85
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 07390c7f-b004-4848-8b9c-f0a991057540
begin
	Pkg.add("CairoMakie")
	Pkg.add("WGLMakie")
end

# ╔═╡ 0e683651-b119-401c-9142-bb9a2a7e56c4
begin
	using Revise, RunwayLib, Plots
	using Unitful.DefaultSymbols, Rotations
	using Distributions, Unitful, Plots
	using LinearAlgebra
	using WGLMakie
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
	noisy_observations = [p + ProjectionPoint(2.0*randn(2)RunwayLib.px) for p in true_observations]
	
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
# ╠═╡ disabled = true
#=╠═╡
begin
	check_calibration("x", 500, 0.01)
end
  ╠═╡ =#

# ╔═╡ 56b8a70b-cd35-49ac-9927-567e38db6b2a
# ╠═╡ disabled = true
#=╠═╡
check_calibration("y", 500, 0.01)
  ╠═╡ =#

# ╔═╡ 85a43112-ff09-439a-a0c1-ef31718616a2
# ╠═╡ disabled = true
#=╠═╡
check_calibration("z", 500, 0.01)
  ╠═╡ =#

# ╔═╡ 081334b7-d1ce-46f9-8555-92f66ce97b29
md"""
## 5. Monte Carlo Integrity Check: Method 1

1. Sample 1000 times. 
2. Filter out the pose estimates that pass the fault detection.
3. Check for the max location differences.
"""

# ╔═╡ f0cafa47-3f83-4fa1-bcfc-e3ecf1e49318
begin

	num = 1000

	# Create runway scenario
	runway_corners = [
		WorldPoint(0.0m, 50m, 0m),     # near left
		WorldPoint(3000.0m, 50m, 0m),  # far left
		WorldPoint(3000.0m, -50m, 0m),  # far right
		WorldPoint(0.0m, -50m, 0m),    # near right
	]
	
	cam_pos = WorldPoint(-2000.0m, 12m, 150m)
	cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)
	
	true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]

	# Store pose estimates as a list of tuples (cam_pos_est, cam_rot_est, stats)
	poses_stats = []

	# Generate pose estimates with statistics
	for i in 1:num_poses
		
		# Make noisy projections
		noisy_observations = [p + ProjectionPoint(2.0*randn(2)RunwayLib.px) for p in true_observations]

		# Estimate pose from noisy projections
		(cam_pos_est, cam_rot_est) = estimatepose6dof(
			PointFeatures(runway_corners, noisy_observations)
		)[(:pos, :rot)]

		# Create noise covariance
		noise_level = 2.0
		sigmas = noise_level * ones(length(runway_corners))
		noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

		# Compute stats
		stats = compute_integrity_statistic(
		    cam_pos_est, cam_rot_est,
		    runway_corners,
		    noisy_observations,
		    noise_cov::Union{<:AbstractMatrix, <:NoiseModel},
		    CAMERA_CONFIG_OFFSET
		)

		push!(poses_stats, (cam_pos_est, cam_rot_est, stats))

	end
end

# ╔═╡ 7f3146d2-fc74-401b-8065-a722f3829341
begin
	
	function plot_poses(poses_stats; world_points=[], color=:blue)
		fig = Figure(size=(600, 500))
		ax = Axis3(fig[1, 1],
			title = "Camera Pose Visualization ($(length(poses_stats)) samples)",
			aspect = (1, 1, 1),
			xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)"
		)
	
		# Plot runway corners
		if !isempty(world_points)
			X = [ustrip(p.x) for p in world_points]
			Y = [ustrip(p.y) for p in world_points]
			Z = [ustrip(p.z) for p in world_points]
			WGLMakie.scatter!(ax, X, Y, Z, color=:gray, markersize=3)
		end
	
		xs = [ustrip(p[1].x) for p in poses_stats]
		ys = [ustrip(p[1].y) for p in poses_stats]
		zs = [ustrip(p[1].z) for p in poses_stats]

		colors = [p[3].p_value > 0.05 ? color : :red for p in poses_stats]
		WGLMakie.scatter!(ax, xs, ys, zs, color=colors, markersize=3)
	
		x_center, y_center, z_center = mean(xs), mean(ys), mean(zs)
		WGLMakie.limits!(
		    ax,
		    x_center - 100, x_center + 100,
		    y_center - 20,  y_center + 20,
		    z_center - 20,  z_center + 20
		)
		fig
	end

	plot_poses(poses_stats, world_points=runway_corners)
end

# ╔═╡ 13f300cd-4861-4eb6-a9a2-9b49be564d8d
begin

	# Get passing poses without units
	passing_poses = [p[1] for p in poses_stats if p[3].p_value > 0.05]
	passing_poses_strip = [
		Point3f(ustrip(p.x), ustrip(p.y), ustrip(p.z)) 
	for p in passing_poses]

	# Get failing poses without units
	failing_poses = [p[1] for p in poses_stats if p[3].p_value <= 0.05]
	failing_poses_strip = [
		Point3f(ustrip(p.x), ustrip(p.y), ustrip(p.z)) 
	for p in failing_poses]
	
	with_theme(theme_black()) do
	    fig = Figure(size=(600, 500))

		# Slider for alpha transparency
	    sl = Makie.Slider(fig[3,1], range=0:0.01:1.0, startvalue=0.6)
		α = sl.value  # reactive node

		# 3D scene
		ax = Axis3(fig[1,1];
				   title = "Monte Carlo Pose Integrity Results ($(length(poses_stats)) samples)",
				   xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
				   xticklabelsvisible = false,
				   yticklabelsvisible = false,
				   zticklabelsvisible = false,
				   )
		
		WGLMakie.scatter!(
			ax, 
			Point3.(passing_poses_strip), 
			alpha=sl.value,
			label="Passing poses (n = $(length(passing_poses_strip)))")

		WGLMakie.scatter!(
			ax, 
			Point3.(failing_poses_strip), 
			alpha=sl.value,
			label="Failing poses (n = $(length(failing_poses_strip)))")

		fig[2,1] = Legend(fig, ax, "Comparing Poses")
		
		fig
	end
end

# ╔═╡ 53612ea4-9f3d-4c28-82a4-e4b3875ba20d
md"""
## 5. Monte Carlo Integrity Check: Method 2

1. For each corner, for a radius around the corner, check if the detection fails.
2. Stop when one fails
3. Look at the max pose delta that can be caused. 
"""

# ╔═╡ 5b2ddcb4-a672-4633-a7a6-f8d85b89fbbc
begin


	# Corner 1
	corner_index = 1

	# 3 spirals
	a = 0
	b = 5
	n = 200
	θs = range(0, 6π; length=n)

	# Store pose estimates as a list of tuples (cam_pos_est, cam_rot_est, stats, observations)
	poses_stats_corners = []

	for θ in θs

		# Copy original observations each iteration to avoid accumulating noise
        noisy_observations = copy(true_observations)

		# Compute spiral offset (units: pixels)
        offset = (a + b*θ) .* [cos(θ), sin(θ)] .* RunwayLib.px
        noisy_observations[corner_index] += ProjectionPoint(offset...)
		
		# Estimate pose from noisy projections
		(cam_pos_est, cam_rot_est) = estimatepose6dof(
			PointFeatures(runway_corners, noisy_observations)
		)[(:pos, :rot)]

		#Create noise covariance
		noise_level = 2.0
		sigmas = noise_level * ones(length(runway_corners))
		noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

		# Compute stats
		stats = compute_integrity_statistic(
		    cam_pos_est, cam_rot_est,
		    runway_corners,
		    noisy_observations,
		    noise_cov::Union{<:AbstractMatrix, <:NoiseModel},
		    CAMERA_CONFIG_OFFSET
		)

		push!(poses_stats_corners, (cam_pos_est, cam_rot_est, stats, noisy_observations))
	end
end

# ╔═╡ b7ca6dbb-1125-4dfc-979d-d66594fa71af
print(length(poses_stats_corners))

# ╔═╡ 1eaa7bcc-f45f-4ff4-935e-5f8f646b29b5
begin
	passing = [p for p in poses_stats_corners if p[3].p_value > 0.05]
	failing = [p for p in poses_stats_corners if p[3].p_value ≤ 0.05]

	passing_points = [
	    Point3f(ustrip(p[1].x), ustrip(p[1].y), ustrip(p[1].z))
	    for p in passing
	]
	failing_points = [
	    Point3f(ustrip(p[1].x), ustrip(p[1].y), ustrip(p[1].z))
	    for p in failing
	]
	
	with_theme(theme_black()) do
	    fig = Figure(size=(700, 600))
	    ax = Axis3(fig[1,1];
	        title = "Spiral Perturbation — Pose Integrity Check",
	        xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
	        xticklabelsvisible = false, yticklabelsvisible = false, zticklabelsvisible = false
	    )
	
	    if !isempty(passing_points)
	        WGLMakie.scatter!(
				ax, 
				passing_points; 
				color=:cyan, 
				markersize=5, 
				label="Passing (p > 0.05)"
			)
	    end
	    if !isempty(failing_points)
	        WGLMakie.scatter!(
				ax,
				failing_points;
				color=:red,
				markersize=5,
				label="Failing (p ≤ 0.05)"
			)
	    end
	
	    fig[2,1] = Legend(fig, ax, "Pose Integrity Results")
	    fig
	end
end

# ╔═╡ ab088db3-48ce-47b6-93ee-6057c7a13001
begin
	with_theme(theme_black()) do
	    fig = Figure(size=(500, 500))
	    ax = Axis(
	        fig[1, 1];
	        title = "Spiral Path of Corner $(corner_index)",
	        xlabel = "x (px)",
	        ylabel = "y (px)",
	        backgroundcolor = :black,
	        xgridvisible = false,
	        ygridvisible = false,
	        xticklabelcolor = :white,
	        yticklabelcolor = :white,
	        xlabelcolor = :white,
	        ylabelcolor = :white,
	        titlecolor = :white,
	    )
	
	    # Extract corner projections and their pass/fail stats
	    corner_spiral = [p[4][corner_index] for p in poses_stats_corners]
	    xs = [ustrip(pp.x) for pp in corner_spiral]
	    ys = [ustrip(pp.y) for pp in corner_spiral]
	    pvals = [p[3].p_value for p in poses_stats_corners]
	
	    # Split into passing / failing groups
	    xs_pass = [x for (x, pv) in zip(xs, pvals) if pv > 0.05]
	    ys_pass = [y for (y, pv) in zip(ys, pvals) if pv > 0.05]
	    xs_fail = [x for (x, pv) in zip(xs, pvals) if pv ≤ 0.05]
	    ys_fail = [y for (y, pv) in zip(ys, pvals) if pv ≤ 0.05]
	
	    # Draw spiral path
	    #lines!(ax, xs, ys; color = :cyan, linewidth = 2)
	
	    # Scatter: passing in cyan, failing in red
    	WGLMakie.scatter!(ax, xs_pass, ys_pass; color = :cyan, markersize = 5, label = "Passing")
    	WGLMakie.scatter!(ax, xs_fail, ys_fail; color = :red, markersize = 5, label = "Failing")

		axislegend(ax; position = :lt, labelcolor = :white, framevisible = false)
	
	    fig
	end

end

# ╔═╡ Cell order:
# ╟─f6f880a7-5c6b-427a-8a0e-a1adf80419ec
# ╟─86dad518-9b91-4dad-ac41-f1ca9409d5e5
# ╟─6072f960-54e8-4a9b-9400-5462fd2eef00
# ╟─e486d0f7-4c49-4fc2-85da-1e149be5ae85
# ╟─07390c7f-b004-4848-8b9c-f0a991057540
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
# ╠═081334b7-d1ce-46f9-8555-92f66ce97b29
# ╠═f0cafa47-3f83-4fa1-bcfc-e3ecf1e49318
# ╠═7f3146d2-fc74-401b-8065-a722f3829341
# ╠═13f300cd-4861-4eb6-a9a2-9b49be564d8d
# ╠═53612ea4-9f3d-4c28-82a4-e4b3875ba20d
# ╠═5b2ddcb4-a672-4633-a7a6-f8d85b89fbbc
# ╠═b7ca6dbb-1125-4dfc-979d-d66594fa71af
# ╠═1eaa7bcc-f45f-4ff4-935e-5f8f646b29b5
# ╠═ab088db3-48ce-47b6-93ee-6057c7a13001
