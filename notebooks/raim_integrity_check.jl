### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 3554a19f-ceec-48d9-ad4d-dd3cc7a37f14
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 22ab9ac9-7ab7-4612-954a-bbd46a20b75a
begin
	Pkg.add("CairoMakie")
	Pkg.add("WGLMakie")
	Pkg.add("StaticArrays")
end

# ╔═╡ 608ec2b2-5f80-48db-95e2-88b564a3ab7f
begin
	using Revise, RunwayLib, Plots
	using Unitful.DefaultSymbols, Rotations
	using Distributions, Unitful, Plots
	using LinearAlgebra
	using WGLMakie
end

# ╔═╡ 822563ea-1e9f-4d0f-a766-a684ac5283b7
md"""
# Integrity Analysis for Runway Pose Estimation

*By Olivia Beyer Bruvik, Stanford University*

This notebook evaluates how noisy or corrupted runway corner detections affect
camera pose estimation. The goal is to quantify **how large an undetected pose
error can be** under a residual-based fault detector.

We use two methods:

### Method 1 — Monte Carlo Sampling
Add random pixel noise to all corner detections, estimate the pose, evaluate the
residual-based p-value, keep passing samples (p > 0.05), and measure the maximum
undetected position error.

### Method 2 — Spiral Corner Perturbation
Perturb a single corner along a 2D spiral, record which offsets pass/fail, and
observe how the pose shifts along the entire spiral. This reveals the structure
of the pass/fail boundary and the estimator’s sensitivity to localized faults.

### RAIM-inspired theoretical approach
We also examine the **worst-case fault direction** using a RAIM-style failure
mode slope formulation, which provides an analytic upper bound on undetected
position bias.

Together, these approaches show how pixel-level faults propagate to pose errors
and when the detector succeeds or fails to flag them.
"""

# ╔═╡ c64fe80a-bf45-11f0-939e-07bfe344d12f
md"""
## Monte Carlo Integrity Check: Method 1

This method performs a brute-force Monte Carlo search to estimate the maximum
position error that can occur **while the residual-based fault detector still passes**.

1. Generate a large number of noisy observations (e.g., 1000 samples).
2. Run the pose estimator and compute the RAIM-like residual statistic for each sample.
3. Keep only the pose estimates whose p-value is above the detection threshold (i.e., faults that go **undetected**).
4. Compute the maximum absolute deviation in the estimated position `(Δx, Δy, Δz)` relative to the true camera position.

This provides an empirical estimate of the **worst-case undetected pose error** under the assumed noise model.
"""

# ╔═╡ 3c0463b0-803c-4c25-8834-9c5d2f4bf661
pwd()

# ╔═╡ cda48361-ba0b-4737-993b-9bc58c342c05
function getPosesStats(num_poses = 1000)

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

		# TODO: try with 3DOF (consider known rotation)
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
		    noise_cov,
		    CAMERA_CONFIG_OFFSET
		)

		push!(poses_stats, (cam_pos_est, cam_rot_est, stats))

	end
	
	return poses_stats, runway_corners, cam_pos
end

# ╔═╡ aaa21c7a-24c5-4cf1-9342-a3e14bde650a
function getPosesStats3DOF(num_poses = 5000, projection_noise=2.0)

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
		noisy_observations = [p + ProjectionPoint(projection_noise*randn(2)RunwayLib.px) for p in true_observations]

		# Estimate pose from noisy projections
		(cam_pos_est, cam_rot_est) = estimatepose3dof(
			PointFeatures(runway_corners, noisy_observations), NO_LINES, cam_rot, 
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
		    noise_cov,
		    CAMERA_CONFIG_OFFSET
		)

		push!(poses_stats, (cam_pos_est, cam_rot_est, stats))

	end
	
	return poses_stats, runway_corners, cam_pos
end

# ╔═╡ 06f94f8d-cb17-4cf9-8351-f27c216875b6
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

# ╔═╡ 0aa65cb8-64ba-4a73-b093-026c636bfbbd
function plotPoseIntegrityResults(poses_stats)

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
			passing_poses_strip, 
			alpha=sl.value,
			label="Passing poses (n = $(length(passing_poses_strip)))")

		WGLMakie.scatter!(
			ax, 
			failing_poses_strip, 
			alpha=sl.value,
			label="Failing poses (n = $(length(failing_poses_strip)))")

		fig[2,1] = Legend(fig, ax, "Comparing Poses")
		
		fig
	end
end

# ╔═╡ 04338b2d-878d-4a69-818c-26a48a10cc92
poses_stats, runway_corners, cam_pos = getPosesStats3DOF()

# ╔═╡ b83e4659-d4a0-4726-bbd6-4a824ffbcec9
# ╠═╡ show_logs = false
plot_poses(poses_stats, world_points=runway_corners)

# ╔═╡ 6df595af-1e0b-41f4-b6a2-50f1dfd6cee9
plotPoseIntegrityResults(poses_stats)

# ╔═╡ 8f59e2ff-ffa5-4520-ad94-95006de8f397
md"""
We want to get the max delta in pose estimation, so we look at the maximum delta in each direction from the cam_pos defined in getPosesStats().

We consider the maximum delta in x, y, z directions.
"""

# ╔═╡ 35bcca17-c6ae-45be-9519-42ddaf17946d
function getMaxXYZ(poses_stats, cam_pos)

	# Get x, y, z values that pass
	xs = [p[1].x for p in poses_stats if p[3].p_value > 0.05]
	ys = [p[1].y for p in poses_stats if p[3].p_value > 0.05]
	zs = [p[1].z for p in poses_stats if p[3].p_value > 0.05]

	# Subtract cam pos values from xs, ys, zs
	x_deltas = ustrip.(xs .- cam_pos.x)
	y_deltas = ustrip.(ys .- cam_pos.y)
	z_deltas = ustrip.(zs .- cam_pos.z)

	# Get max values
	max_xi = argmax(abs.(x_deltas))
	max_yi = argmax(abs.(y_deltas))
	max_zi = argmax(abs.(z_deltas))

	return x_deltas[max_xi], y_deltas[max_yi], z_deltas[max_zi]
	
end

# ╔═╡ 80f1c84c-a035-4584-b84f-e7110ac748c5
begin
	Δx, Δy, Δz  = getMaxXYZ(poses_stats, cam_pos)

	println("Max Δx = ", Δx, " m")
	println("Max Δy = ", Δy, " m")
	println("Max Δz = ", Δz, " m")
end

# ╔═╡ 7fbfd4ad-6c9d-494a-a6c4-fbafbb325b7c
md"""
## Monte Carlo Integrity Check: Method 2 (Spiral Corner Perturbation)

This method systematically sweeps through a spiral of pixel perturbations applied to a single runway corner. Instead of stopping at the first failure, the method traces an
entire spiral of offsets, allowing us to observe **where in pixel space the
residual test begins to fail**.

1. For a selected corner, generate a 2D spiral of increasing pixel offsets
   (covering multiple radii and directions).
2. For each perturbed observation, run the pose estimator and evaluate the
   RAIM-like residual test statistic.  
   This produces a full map of **passing** and **failing** perturbations,
   rather than stopping at the first failing case.
3. Record the pose estimate for every point along the spiral to visualize how
   the camera position changes under localized corner faults.

This gives a complete picture of the **fault sensitivity** of the pose estimator
for that corner: regions of pixel space where perturbations remain undetected
versus regions that trigger the detector. It complements Method 1 by showing the
**structure of the pass/fail boundary** rather than just the maximum undetected
bias.
"""

# ╔═╡ e0fb9177-65ad-43b8-adbd-2e54bc273db2
corner_index = 1

# ╔═╡ a6e2e410-cd45-43b9-9653-40fd57966fee
function getPosesStatsCorners(corner_index=1)

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

	# Spiral with 6 turns
	a = 0
	b = 2
	n = 600
	θs = range(0, 12π; length=n)

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
		    noise_cov,
		    CAMERA_CONFIG_OFFSET
		)

		push!(poses_stats_corners, (cam_pos_est, cam_rot_est, stats, noisy_observations))
	end

	return poses_stats_corners
end

# ╔═╡ 98de9a80-828d-4f5f-a4cc-d1c841c45ffd
function plotSpiralPoses(poses_stats_corners, corner_index)

	# Get passing and failing samples
	passing_samples = [p for p in poses_stats_corners if p[3].p_value > 0.05]
	failing_samples = [p for p in poses_stats_corners if p[3].p_value <= 0.05]

	passing_points = [
	    Point3f(ustrip(p[1].x), ustrip(p[1].y), ustrip(p[1].z))
	    for p in passing_samples
	]
	failing_points = [
	    Point3f(ustrip(p[1].x), ustrip(p[1].y), ustrip(p[1].z))
	    for p in failing_samples
	]

	# Create plot
	with_theme(theme_black()) do
	    fig = Figure(size=(700, 600))
	    ax = Axis3(fig[1,1];
	        title = "Spiral Perturbation — Pose Integrity Check",
	        xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
	        #xticklabelsvisible = false, yticklabelsvisible = false, zticklabelsvisible = false
			aspect = :data #(1, 1, 1)
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
				label="Failing (p <= 0.05)"
			)
	    end
	
	    fig[2,1] = Legend(fig, ax, "Pose Integrity Results")
	    fig
	end
end

# ╔═╡ f89992f2-ea57-4c35-8191-31d960e1f770
function plotSpiralCorners(poses_stats_corners, corner_index=1)
	
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
	    xs_fail = [x for (x, pv) in zip(xs, pvals) if pv <= 0.05]
	    ys_fail = [y for (y, pv) in zip(ys, pvals) if pv <= 0.05]
	
	    # Draw spiral path
	    #lines!(ax, xs, ys; color = :cyan, linewidth = 2)
	
	    # Scatter: passing in cyan, failing in red
    	WGLMakie.scatter!(ax, xs_pass, ys_pass; color = :cyan, markersize = 5, label = "Passing")
    	WGLMakie.scatter!(ax, xs_fail, ys_fail; color = :red, markersize = 5, label = "Failing")

		axislegend(ax; position = :lt, labelcolor = :white, framevisible = false)
	
	    fig
	end

end

# ╔═╡ 9345a90d-82b7-4866-8a2a-10f2b2ff4a3b
begin
	poses_stats_corners = getPosesStatsCorners(corner_index)
	plotSpiralPoses(poses_stats_corners, corner_index)
end

# ╔═╡ 4bf4bbd5-2d82-4630-a3a6-2ba479bffe2d
plotSpiralCorners(poses_stats_corners, corner_index)

# ╔═╡ 0f0d6f98-56e5-4cea-bf34-bdf6e3b4103b
let
	Δx, Δy, Δz  = getMaxXYZ(poses_stats, cam_pos)

	println("Max Δx = ", Δx, " m")
	println("Max Δy = ", Δy, " m")
	println("Max Δz = ", Δz, " m")
end

# ╔═╡ f938f8ef-15cc-4537-be65-83539e9ad771
md"""

## RAIM-Inspired Integrity Analysis (Failure Mode Slope Formulation)

**Goal:**  
Find the fault direction $f$ that produces the **maximum position-domain error**
while still satisfying the **residual-based fault detection constraint**.

$\max_{f} \; |S_0^T f|$

subject to

$f^T (I_n - H S_0)\, f \le T_{RB}^2$,

where $T_{RB}$ is the residual-based detection threshold and  
$S_0$ is the sensitivity (or position-domain mapping) matrix.

---

### Definition of the position-domain sensitivity vector

$s_0 = \alpha S_0 = \alpha H^{+}, \qquad \alpha = [0\;\;0\;\;1\;\;0].$

Here, $\alpha$ selects the state component of interest (e.g., vertical error).

---

### Failure Mode Slope

The *failure mode slope* measures how strongly a given fault direction $f$  
affects the position estimate relative to its detectability:

$g_F^2 \equiv \frac{f^T S_0 S_0^T f}{f^T (I_n - H S_0)\, f}$

Large $g_F$ means the fault generates a large position error while remaining  
hard to detect by the residual test.

---

### Worst-Case Failure Mode Slope

The worst case occurs when the constraint is tight and the fault aligns with
the most harmful direction:

$\bar{g}_{F,i}^2 = s_0^T (I_n - H S_0)^{-1} s_0$

This gives the maximum possible failure mode slope for component $i$.

---

### Worst-Case Fault Direction

The corresponding maximally harmful fault vector is

$\bar{f}_i = (I_n - H S_0)^{-1} S_0 = \sigma\, \Delta_i$

where $\Delta_i$ is the unit direction of the worst-case mode and $\sigma$ 
scales the fault to the residual threshold.

---

### Worst-Case Position Error

Finally, the worst-case undetected position-domain error can be found using line search. This represents the **largest bias in position** that can escape detection for the chosen state dimension.
"""


# ╔═╡ 65b4a3aa-37e7-4668-989d-f404f18df951
let

	# Get x, y, z values that pass
	xs = [p[1].x for p in poses_stats]
	ys = [p[1].y for p in poses_stats]
	zs = [p[1].z for p in poses_stats]
	pvalues = [p[3].p_value for p in poses_stats]

	# Subtract cam pos values from xs, ys, zs
	x_deltas = ustrip.(xs .- cam_pos.x)
	y_deltas = ustrip.(ys .- cam_pos.y)
	z_deltas = ustrip.(zs .- cam_pos.z)

	# === Plot results in dark mode ===
    with_theme(theme_black()) do
        fig = Figure(size = (600, 400))
        ax = Axis(fig[1, 1],
		    title = "p-value of Residual Test vs Position Error",
		    xlabel = "Error (m)",
		    ylabel = "p-value (p-value < 0.05 fails)",
		    backgroundcolor = :black,
		    xgridvisible = false,
		    ygridvisible = false,
		)

        WGLMakie.scatter!(ax, x_deltas, pvalues; color = :cyan, markersize = 6)
        fig
    end

end
	

# ╔═╡ 27c8fbc0-c8b5-4d21-8cb5-c70a121343be
let
    xs = [p[1].x for p in poses_stats]
    pvalues = [p[3].p_value for p in poses_stats]

    x_deltas = ustrip.(xs .- cam_pos.x)
    passes = pvalues .> 0.05

    # Adaptive bin range based on data
    min_x = floor(minimum(x_deltas))
    max_x = ceil(maximum(x_deltas))
    bin_edges = collect(min_x:10:max_x)
    bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

    # Compute pass rates
    pass_rate = Float64[]
    std_rate = Float64[]
    for i in 1:length(bin_centers)
        in_bin = findall(x -> x >= bin_edges[i] && x < bin_edges[i+1], x_deltas)
        if isempty(in_bin)
            push!(pass_rate, NaN)
            push!(std_rate, NaN)
        else
            r = mean(passes[in_bin])
            σ = √(r*(1-r)/length(in_bin))
            push!(pass_rate, r)
            push!(std_rate, σ)
        end
    end

    # Smooth
    smooth_rate = [mean(skipmissing(pass_rate[max(1,i-1):min(end,i+1)])) for i in 1:length(pass_rate)]
    smooth_std  = [mean(skipmissing(std_rate[max(1,i-1):min(end,i+1)])) for i in 1:length(std_rate)]

    with_theme(theme_black()) do
        fig = Figure(size = (700, 500))

        # Upper plot — p-values
        ax1 = Axis(fig[1, 1],
            title = "p-value vs Error",
            xlabel = "Error (m)",
            ylabel = "p-value (p-value < 0.05 fails)",
            backgroundcolor = :black)

        WGLMakie.scatter!(ax1, x_deltas, pvalues; color = :cyan, markersize = 4)

        # Lower plot — detection probability
        ax2 = Axis(fig[2, 1],
            title = "Detection Probability vs Error",
            xlabel = "Error (m)",
            ylabel = "Probability of passing",
            backgroundcolor = :black)

        lower = clamp.(smooth_rate .- smooth_std, 0, 1)
        upper = clamp.(smooth_rate .+ smooth_std, 0, 1)

        WGLMakie.band!(ax2, bin_centers, lower, upper; color = (:lime, 0.25))
        WGLMakie.lines!(ax2, bin_centers, smooth_rate; color = :lime, linewidth = 2, label = "Pass Rate")
        WGLMakie.lines!(ax2, bin_centers, 1 .- smooth_rate; color = :magenta, linewidth = 2, label = "Detection Probability")

        axislegend(ax2)

        fig
    end
end


# ╔═╡ Cell order:
# ╟─822563ea-1e9f-4d0f-a766-a684ac5283b7
# ╟─c64fe80a-bf45-11f0-939e-07bfe344d12f
# ╟─3c0463b0-803c-4c25-8834-9c5d2f4bf661
# ╟─3554a19f-ceec-48d9-ad4d-dd3cc7a37f14
# ╠═22ab9ac9-7ab7-4612-954a-bbd46a20b75a
# ╟─608ec2b2-5f80-48db-95e2-88b564a3ab7f
# ╟─cda48361-ba0b-4737-993b-9bc58c342c05
# ╟─aaa21c7a-24c5-4cf1-9342-a3e14bde650a
# ╟─06f94f8d-cb17-4cf9-8351-f27c216875b6
# ╟─0aa65cb8-64ba-4a73-b093-026c636bfbbd
# ╠═04338b2d-878d-4a69-818c-26a48a10cc92
# ╠═b83e4659-d4a0-4726-bbd6-4a824ffbcec9
# ╠═6df595af-1e0b-41f4-b6a2-50f1dfd6cee9
# ╟─8f59e2ff-ffa5-4520-ad94-95006de8f397
# ╟─35bcca17-c6ae-45be-9519-42ddaf17946d
# ╟─80f1c84c-a035-4584-b84f-e7110ac748c5
# ╟─7fbfd4ad-6c9d-494a-a6c4-fbafbb325b7c
# ╟─e0fb9177-65ad-43b8-adbd-2e54bc273db2
# ╟─a6e2e410-cd45-43b9-9653-40fd57966fee
# ╟─98de9a80-828d-4f5f-a4cc-d1c841c45ffd
# ╟─f89992f2-ea57-4c35-8191-31d960e1f770
# ╠═4bf4bbd5-2d82-4630-a3a6-2ba479bffe2d
# ╟─9345a90d-82b7-4866-8a2a-10f2b2ff4a3b
# ╟─0f0d6f98-56e5-4cea-bf34-bdf6e3b4103b
# ╟─f938f8ef-15cc-4537-be65-83539e9ad771
# ╟─65b4a3aa-37e7-4668-989d-f404f18df951
# ╟─27c8fbc0-c8b5-4d21-8cb5-c70a121343be
