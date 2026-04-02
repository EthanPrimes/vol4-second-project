"""
Single-shooting solver for the minimum-time sailing BVP.
Julia port of shooting.py — uses OrdinaryDiffEq + NLsolve + ForwardDiff.

Unknowns: (λ₁, λ₂(0), T) — 3 values, 3 conditions:
    x(1) = xf[1],  y(1) = xf[2],  H(t=1) = 0
"""

using OrdinaryDiffEq
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots

# ---------------------------------------------------------------------------
# PLUG AND PLAY: swap out phi to change the wind field.
# phi(x, y) -> SVector or 2-element vector [wind_x, wind_y].
# Uses plain Julia arithmetic so ForwardDiff can differentiate through it.
# Examples:
#   constant:       [-1.0, -1.0]
#   linear shear:   [-1.0 - 0.1*y, -1.0]
#   vortex:         [-sin(x), cos(y)]
# ---------------------------------------------------------------------------
phi(x, y) = [-1.0, -1.0]

# ---------------------------------------------------------------------------
# Sailing polar: boat speed as a function of heading and wind field phi(x,y).
# ---------------------------------------------------------------------------
function V(heading, x, y)
    wvec     = phi(x, y)
    wind_vel = norm(wvec)
    wind_dir = atan(wvec[2], wvec[1]) - π
    alpha    = mod(heading - wind_dir + π, 2π) - π
    no_go    = deg2rad(45)
    abs(alpha) < no_go && return zero(heading)   # no-go zone; zero() preserves type for AD
    speed_mult = sin(alpha)^2 * (1 + cos(alpha)^2) / 2
    return wind_vel * speed_mult
end

# Gradient of V w.r.t. (x, y) via ForwardDiff, so it works for any differentiable phi.
function grad_V_xy(heading, x, y)
    f = xy -> V(heading, xy[1], xy[2])
    ForwardDiff.gradient(f, [x, y])
end

# ---------------------------------------------------------------------------
# Ocean current w(x,y) and its Jacobian dw/d(x,y).
# ---------------------------------------------------------------------------
w(x, y)  = [0.0, 0.0]
dw(x, y) = [0.0  0.0;
             0.0  0.0]

# ---------------------------------------------------------------------------
# Hamiltonian (negated so we minimise with standard optimisers).
# H = λ·(v + w) − 1
# ---------------------------------------------------------------------------
function neg_H(theta, x, y, l1, l2)
    v_speed = V(theta, x, y)
    v1 = v_speed * cos(theta)
    v2 = v_speed * sin(theta)
    w1, w2 = w(x, y)
    -(l1 * (v1 + w1) + l2 * (v2 + w2) - 1)
end

# ---------------------------------------------------------------------------
# Global argmax of H over heading: coarse grid + Brent refinement.
# Grid search avoids local-minima traps from the multi-modal sailing polar.
# ---------------------------------------------------------------------------
function optimal_theta(x, y, l1, l2; n_grid=64)
    thetas = range(-π, π, length=n_grid+1)[1:end-1]
    best   = thetas[argmin(neg_H.(thetas, x, y, l1, l2))]

    # Brent's method on a bracket around best grid point
    dtheta = 2π / n_grid
    lo, hi = best - dtheta, best + dtheta
    result = optimize(t -> neg_H(t, x, y, l1, l2), lo, hi, Brent())
    return result.minimizer
end

# Optim.jl needed for Brent — import here to keep it local
using Optim: optimize, Brent

# ---------------------------------------------------------------------------
# ODE right-hand side (unscaled time on [0,1], T is a parameter).
# State: Y = [x, y, λ₁, λ₂]
# ---------------------------------------------------------------------------
function ode!(dY, Y, T, t)
    x, y, l1, l2 = Y
    theta = optimal_theta(x, y, l1, l2)

    v_speed   = V(theta, x, y)
    v_vec     = v_speed .* [cos(theta), sin(theta)]
    w_vec     = w(x, y)
    direction = [cos(theta), sin(theta)]
    lam       = [l1, l2]
    dV        = grad_V_xy(theta, x, y)

    dl  = -T .* (dot(lam, direction) .* dV .+ dw(x, y)' * lam)
    dxy = T .* (v_vec .+ w_vec)

    dY[1] = dxy[1]
    dY[2] = dxy[2]
    dY[3] = dl[1]
    dY[4] = dl[2]
end

# ---------------------------------------------------------------------------
# Shooting residual: integrate forward, return BC errors.
# params = [λ₁, λ₂(0), T]
# ---------------------------------------------------------------------------
const x0 = [0.0, 0.0]
const xf = [4.0, 4.0]

function shoot(params)
    l1, l2_0, T = params
    T <= 0 && return fill(1e6, 3)

    Y0  = [x0[1], x0[2], l1, l2_0]
    prob = ODEProblem(ode!, Y0, (0.0, 1.0), T)
    sol  = solve(prob, RK4(); dt=5e-3, adaptive=true, reltol=1e-8, abstol=1e-10)

    x_end, y_end, l1_end, l2_end = sol.u[end]
    theta_f = optimal_theta(x_end, y_end, l1_end, l2_end)
    H_f     = -neg_H(theta_f, x_end, y_end, l1_end, l2_end)

    [x_end - xf[1], y_end - xf[2], H_f]
end

# ---------------------------------------------------------------------------
# Initial guess (same reasoning as shooting.py).
# λ̇₁ = 0 → λ₁ constant.   λ̇₂ = −T·λ₁ → λ₂ linear.
# H(tf)=0 at y=xf[2] with V≈0 → λ₁·xf[2] ≈ 1.
# ---------------------------------------------------------------------------
T_guess  = 8.0
l1_guess = 1.0 / (xf[2] + 1.5)
l2_guess = 0.3

println("Residual at initial guess: ", shoot([l1_guess, l2_guess, T_guess]))

result = nlsolve(shoot, [l1_guess, l2_guess, T_guess]; method=:trust_region, ftol=1e-8)
println("Converged: ", converged(result))
println("Residual:  ", result.residual_norm)
println("(λ₁, λ₂₀, T): ", result.zero)

converged(result) || error("Shooting failed — adjust initial guess.")

l1_sol, l2_sol, T_sol = result.zero

# ---------------------------------------------------------------------------
# Reconstruct trajectory at fine resolution for plotting.
# ---------------------------------------------------------------------------
Y0   = [x0[1], x0[2], l1_sol, l2_sol]
prob = ODEProblem(ode!, Y0, (0.0, 1.0), T_sol)
traj = solve(prob, RK4(); dt=1e-3, adaptive=true, reltol=1e-10, abstol=1e-12)

t_plot = range(0, 1, length=500)
Y_plot = hcat(traj.(t_plot)...)   # 4 × 500

xs_path = Y_plot[1, :]
ys_path = Y_plot[2, :]
thetas  = [optimal_theta(Y_plot[1,i], Y_plot[2,i], Y_plot[3,i], Y_plot[4,i])
           for i in eachindex(t_plot)]

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
margin = 0.5
gx = range(x0[1] - margin, xf[1] + margin, length=10)
gy = range(x0[2] - margin, xf[2] + margin, length=10)
Xg = [xi for xi in gx, yi in gy]
Yg = [yi for xi in gx, yi in gy]
Ug = fill(-1.0, size(Xg))
Vg = fill(-1.0, size(Yg))

p1 = quiver(Xg[:], Yg[:], quiver=(Ug[:] .* 0.15, Vg[:] .* 0.15),
            color=:gray, alpha=0.5, aspect_ratio=:equal,
            xlabel="x", ylabel="y", title="Optimal path  (T = $(round(T_sol, digits=3)))",
            legend=:topright)
plot!(p1, xs_path, ys_path, lw=2, color=:blue, label="path")
scatter!(p1, [x0[1]], [x0[2]], ms=8, color=:green, label="start")
scatter!(p1, [xf[1]], [xf[2]], ms=8, color=:red,   label="end")

p2 = plot(t_plot .* T_sol, rad2deg.(thetas),
          xlabel="time", ylabel="heading (deg)",
          title="Optimal heading θ*(t)", legend=false)

plot(p1, p2, size=(1100, 480))
savefig("shooting_solution.png")
println("Plot saved to shooting_solution.png")
