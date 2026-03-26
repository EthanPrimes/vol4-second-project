# Derivation of Solution for Constant Wind Navigation

## Set up
For a sailboat moving in a place with fixed wind speed and direction, we only care about the heading. Our control, then, is $\theta$. For this problem we assume that the sail is fixed in place, but we can change the direction in which the boat points. We also assume that the wind directly controls the velocity, instead of the acceleration.

Let $v$ refer to the speed of the wind, with $\phi$ the angle at which the wind is blowing in radians, where $\phi = 0$ corresponding to wind blowing in the east direction.

Our state $\vec{s}$ is given by $$ \begin{bmatrix} x \\ y \end{bmatrix}$$ which are simply our $x$ and $y$ coordinates. 

The control vector, or $\vec{u}$, is given by the $x$ and $y$ components of the fixed velocity $v$ in the direction of $\theta,$ multiplied by the exposure of the sail to the wind:

$$ \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = v \begin{bmatrix} \cos(\theta - \phi) \cos(\theta) \\ \cos(\theta - \phi) \sin(\theta) \end{bmatrix} $$

The velocity vector is the control vector plus the effects of the drift $w$:

$$ \dot{\begin{bmatrix} x \\ y \end{bmatrix}} = v \begin{bmatrix} \cos(\theta - \phi) \cos(\theta) \\ \cos(\theta - \phi) \sin(\theta) \end{bmatrix} + \begin{bmatrix} w_1(x, y) \\ w_2(x, y) \end{bmatrix}$$

This problem's constraints are fixed endpoints and a norm for our control $u$.
$$ \| u \| = v \cos^2(\theta - \phi) \\ \vec{s}(t_0) = \vec{s}_0 \\ \vec{s}(t_f) = \vec{s}_f $$

## Optimization problem
We have 
$$ J[u] = \int_{t_0}^{t_f} 1 dt \\ \text{subject to } \dot{\vec{s}} = \vec{u}(t) + \vec{w}(\vec{s}(t)) \\ \vec{s}(t_0) = \vec{s}_0 \\ \vec{s}(t_f) = \vec{s}_f \\ \| u \| = v \cos^2(\theta - \phi)$$

So the lagrangian constraint version is 

$$ \tilde{J}[u] = \int_{t_0}^{t_f} 1 + \lambda^T (\dot{\vec{s}} - \vec{u}(t) - \vec{w}(\vec{s}(t))) dt $$

And so

$$
\begin{aligned}
H &= \lambda^\top f - L \\
  &= \begin{bmatrix} \lambda_1 & \lambda_2 \end{bmatrix}
     \left( v \begin{bmatrix} \cos(\theta - \phi) \cos(\theta) + w_1(x, y) \\ \cos(\theta - \phi) \sin(\theta) + w_2(x, y) \end{bmatrix} \right) - 1
\end{aligned}
$$

By Pontryagin's Maximization principle, we have

$$ \begin{aligned}
\dot{x} &= v\cos(\theta - \phi) \cos(\theta) + w_1(x, y) \\
\dot{y} &= v\cos(\theta - \phi) \sin(\theta) + w_2(x, y) \\
\dot{\lambda_1} &= -\lambda_1 \frac{\partial w_1}{\partial x} -\lambda_2 \frac{\partial w_2}{\partial x} \\
\dot{\lambda_2} &= -\lambda_1 \frac{\partial w_1}{\partial y} -\lambda_2 \frac{\partial w_2}{\partial y} \\
H(t_f) &= 0
\end{aligned}
$$

Where the last equality comes from the fact that we are maximizing over unknown time. We don't get that $\frac{\partial H}{\partial u} = 0$ because $u$ is not unconstrained in this problem.

Since we are maximizing $H$, and the only term where $u$ shows up is $\lambda \cdot u$, we can just maximize that to solve for u.

We get $u = \frac{\lambda}{\| \lambda \|}$ at each time $t$, which we can plug into the above system.

## Implementation

This is implemented in [this notebook](constant_wind.ipynb), though imperfectly because I hardcode the function $w()$ in the `ode()` function. It's just a shape issue. I am currently solving it over the time interval $[0, 1]$, and then solving for $t_f$ at the same time I solve for the optimal route. I pass that in as the argument `p`, and rescale everything in the `ode()` and `bc()` functions accordingly. I will include more here about the change of variables eventually.
