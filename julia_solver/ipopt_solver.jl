using JuMP, Ipopt
import MathOptInterface as MOI
using LinearAlgebra
using Plots
using JLD2
using HDF5

# ---------------------------------------------------------------------------
# PLUG AND PLAY: change phi and NO_GO_DEG, nothing else needs touching.
# phi returns the wind vector [wx, wy] at position (x, y).
# Must be differentiable (no hard branches) for ForwardDiff/Ipopt.
# ---------------------------------------------------------------------------
# Wind speed conversion: mph → degrees/hour
# At lat ~45°N: 1° lat ≈ 69.0 mi, 1° lon ≈ 69.0 * cos(45°) ≈ 48.8 mi
const LAT_CENTER       = 45.1025          # midpoint of 44.990–45.215
const MILES_PER_DEG_LAT = 69.0
const MILES_PER_DEG_LON = 69.0 * cos(deg2rad(LAT_CENTER))  # ≈ 48.8

# Wind components in mph → degrees/hour
const WX_MPH =  2.43   # eastward
const WY_MPH =  4.05   # northward

phi(_, _) = [WX_MPH / MILES_PER_DEG_LON, WY_MPH / MILES_PER_DEG_LAT]

const NO_GO_DEG = 45.0

function solve(x0_val, y0_val, xf_val, yf_val, N, filename, want_plot, plot_title)
    # ---------------------------------------------------------------------------
    # Sailing polar.
    # The hard if-branch of the original causes INVALID_MODEL because ForwardDiff
    # can't compute a valid Hessian at the no-go boundary. Replace with a smooth
    # tanh step: 0 inside no-go zone, 1 outside, ~5° transition width (k=20).
    # ---------------------------------------------------------------------------
    function V(theta, x, y)
        wvec     = phi(x, y)
        wind_vel = sqrt(wvec[1]^2 + wvec[2]^2)
        wind_dir = atan(wvec[2], wvec[1]) - π
        alpha    = mod(theta - wind_dir + π, 2π) - π
        no_go    = NO_GO_DEG * π / 180
        speed_mult   = sin(alpha)^2 * (1 + cos(alpha)^2) / 2
        nogo_weight  = (1 + tanh(50.0 * (abs(alpha) - no_go))) / 2
        return wind_vel * speed_mult * nogo_weight
    end

    # ---------------------------------------------------------------------------
    # Derive initial-guess tack headings automatically from phi.
    # The two optimal tack headings are just outside the no-go boundary on each
    # side of the directly-upwind direction.
    # ---------------------------------------------------------------------------
    function tack_headings()
        w0       = phi((x0_val + xf_val) / 2, (y0_val + yf_val) / 2)
        wind_dir = atan(w0[2], w0[1]) - π
        no_go    = NO_GO_DEG * π / 180
        margin   = deg2rad(5.0)
        upwind   = wind_dir + π
        ta = mod(upwind + no_go + margin, 2π)
        tb = mod(upwind - no_go - margin, 2π)
        return ta, tb
    end

    # ---------------------------------------------------------------------------
    # Build and solve the NLP.
    # ---------------------------------------------------------------------------
    # Use Ipopt, interior point solver for nonlinear problems
    model = Model(Ipopt.Optimizer)

    # Set model attributes like tol and max_iter
    set_optimizer_attribute(model, "print_level", 3)
    set_optimizer_attribute(model, "max_iter", 100000)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "acceptable_tol", 1e-2)

    # Set op_v as the 'alias' of V within the model
    @operator(model, op_V, 3, V)

    # Set variables x, y, theta, and T with built in bounds
    @variable(model, x[1:N])
    @variable(model, y[1:N])
    @variable(model, 0 <= theta[1:N] <= 2π)
    @variable(model, T >= 0.1)

    # Set objective, Mayer with penalty for tacks
    # L1-style penalty: sqrt((Δθ)² + δ) is smooth but charges ~|Δθ| per step.
    # Unlike L2, this strongly penalises many small switches over one large one —
    # the total cost of 100 chatters each of 1° equals the cost of one switch of 100°,
    # so Ipopt strongly prefers to consolidate switching into as few steps as possible.
    # Increase eps to accept a longer T in exchange for cleaner legs.
    eps = 10.0
    smooth = 1e-4   # smoothing; keeps the Hessian nonsingular at Δθ=0
    @objective(model, Min, T + eps * sum(sqrt((theta[i+1]-theta[i])^2 + smooth) for i in 1:N-1))

    # Set starting and ending fixed points
    @constraint(model, x[1] == x0_val)
    @constraint(model, y[1] == y0_val)
    @constraint(model, x[N] == xf_val)
    @constraint(model, y[N] == yf_val)

    # Euler forward dynamics, no current
    # This chunk establishes that x' = f(t, x, u)
    for i in 1:N-1
        @constraint(model, (N - 1) * (x[i+1] - x[i]) == T * op_V(theta[i], x[i], y[i]) * cos(theta[i]))
        @constraint(model, (N - 1) * (y[i+1] - y[i]) == T * op_V(theta[i], x[i], y[i]) * sin(theta[i]))
    end

    # ---------------------------------------------------------------------------
    # Initial guess: straight-line (x,y), theta alternates between the two tack
    # headings every half of the trajectory.  T estimated from distance/speed.
    # ---------------------------------------------------------------------------
    tack_a, tack_b = tack_headings()

    for i in 1:N
        s = (i - 1) / (N - 1)
        set_start_value(x[i], x0_val + s * (xf_val - x0_val))
        set_start_value(y[i], y0_val + s * (yf_val - y0_val))
        set_start_value(theta[i], s < 0.5 ? tack_a : tack_b)
    end

    dist   = sqrt((xf_val - x0_val)^2 + (yf_val - y0_val)^2)
    v_est  = V(tack_a, (x0_val + xf_val) / 2, (y0_val + yf_val) / 2)
    T_init = v_est > 1e-3 ? dist / v_est * 1.5 : 100.0
    set_start_value(T, T_init)

    println("Tack headings (deg): ", round.(rad2deg.([tack_a, tack_b]), digits=1))
    println("T initial guess: ", round(T_init, digits=2))

    optimize!(model)

    println("$plot_title")
    println("Status:    ", termination_status(model))
    println("Min time T: ", value(T))
    println("-" ^ 50)

    x_sol, y_sol, theta_sol, T_sol = value.(x), value.(y), value.(theta), value(T)


    h5open("$filename.h5", "w") do f
        f["x"]     = x_sol
        f["y"]     = y_sol
        f["theta"] = theta_sol
        f["T"]     = T_sol
    end
    println("Saved solution to $filename.h5")


    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    if want_plot
        xs  = range(x0_val, xf_val, length=12)
        ys  = range(y0_val, yf_val, length=12)
        Xg  = [xi for xi in xs, yi in ys]
        Yg  = [yi for xi in xs, yi in ys]
        w0  = phi(15.0, 15.0)
        Ug  = fill(w0[1], size(Xg))
        Vg  = fill(w0[2], size(Yg))

        T_rounded = round(value(T_sol), digits=2)
        p1 = quiver(Xg[:], Yg[:], quiver=(Ug[:] .* 1.5, Vg[:] .* 1.5),
            color=:gray, alpha=0.4, aspect_ratio=:equal,
            xlabel="x", ylabel="y", title="Optimal path $plot_title (T = $T_rounded)",
            legend=:topright)
        plot!(p1, x_sol, y_sol, color=:blue, lw=2, label="path")
        scatter!(p1, [x0_val], [y0_val], color=:green, ms=8, label="start")
        scatter!(p1, [xf_val], [yf_val], color=:red,   ms=8, label="end")

        p2 = plot(range(0, value(T_sol), length=N), rad2deg.(theta_sol),
            xlabel="time", ylabel="heading (deg)",
            title="Heading θ(t)", legend=false)

        plot(p1, p2, size=(1100, 480))
        savefig("ipopt_solution_$filename.png")
        println("Saved to ipopt_solution_$filename.png")
    end

    return value.(x), value.(y), value.(theta), value(T), x0_val, y0_val, xf_val, yf_val, N

end

# Legs of the race
LEGS = [
    ("Leg 1: Menominee to Green Island",        (-87.614, 45.108), (-87.495, 44.990), "leg_1"),
    ("Leg 2: Green Island to Fish Creek",        (-87.495, 44.990), (-87.245, 45.135), "leg_2"),
    ("Leg 3: Fish Creek to Strawberry S",        (-87.245, 45.135), (-87.260, 45.180), "leg_3"),
    ("Leg 4: Strawberry S to Strawberry N",      (-87.260, 45.180), (-87.270, 45.215), "leg_4"),
    ("Leg 5: Strawberry N to Chambers Island",   (-87.270, 45.215), (-87.375, 45.200), "leg_5"),
    ("Leg 6: Chambers Island to Menominee",      (-87.375, 45.200), (-87.614, 45.108), "leg_6"),
]

for (plot_title, start, finish, filename) in LEGS
    x0_val, y0_val = start
    xf_val, yf_val = finish
    solve(x0_val, y0_val, xf_val, yf_val, 150, filename, true, plot_title)
end
