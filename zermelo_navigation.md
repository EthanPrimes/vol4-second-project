# Derivation of Solution for Classic Zermelo Navigation

## Set up
For a motor boat going a fixed speed $v$ in a current drift vector field $w$, we only care about the heading. Our control, then, is $\theta$. 

Our state $\vec{s}$ is given by $$ \begin{bmatrix} x \\ y \end{bmatrix}$$ which are simply our $x$ and $y$ coordinates. 

The control vector, or $\vec{u}$, is given by the $x$ and $y$ components of the fixed velocity $v$ in the direction of $\theta$

$$ \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = v \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} $$

The velocity vector is the control vector plus the effects of the drift $w$:

$$ \dot{\begin{bmatrix} x \\ y \end{bmatrix}} = v \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} + \begin{bmatrix} w_1(x, y) \\ w_2(x, y) \end{bmatrix}$$


This problem's constraints are fixed endpoints and a norm for our control $u$.
$$ \| u \| = v \\ \vec{s}(t_0) = \vec{s}_0 \\ \vec{s}(t_f) = \vec{s}_f $$

## Optimization problem
We have 
$$ J[u] = \int_{t_0}^{t_f} 1 dt \\ \text{subject to } \dot{\vec{s}} = \vec{u}(t) + \vec{w}(\vec{s}(t)) \\ \vec{s}(t_0) = \vec{s}_0 \\ \vec{s}(t_f) = \vec{s}_f \\ \| u \| = v$$

So the lagrangian constraint version is 

$$ \tilde{J}[u] = \int_{t_0}^{t_f} 1 + \lambda^T (\dot{\vec{s}} - \vec{u}(t) - \vec{w}(\vec{s}(t))) dt $$

And so

$$
\begin{aligned}
H &= \lambda^\top f - L \\
  &= \begin{bmatrix} \lambda_1 & \lambda_2 \end{bmatrix}
     \left( v \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} \right) - 1
\end{aligned}
$$

By Potrayigin's Maximization principle, we have

$$ \begin{aligned}
\dot{x} &= v\cos(\theta) + w_1(x, y) \\
\dot{y} &= v\sin(\theta) + w_2(x, y) \\
\dot{\lambda_1} &= -\lambda_1 \frac{\partial w_1}{\partial x} -\lambda_2 \frac{\partial w_2}{\partial x} \\
\dot{\lambda_2} &= -\lambda_1 \frac{\partial w_1}{\partial y} -\lambda_2 \frac{\partial w_2}{\partial y} \\
H(t_f) &= 0
\end{aligned}
$$

Where the last equality comes from the fact that we are maximizing over unknown time. We don't get that $\frac{\partial H}{\partial u} = 0$ because $u$ is not unconstrained in this problem.

Since we are maximizing $H$, and the only term where $u$ shows up is $\lambda \cdot u$, we can just maximize that to solve for u.

We get $u = v \frac{\lambda}{\| \lambda \|}$ at each time $t$, which we can plug into the above system.

## Implementation

This is implemented in [this notebook](zermelo.ipynb), though imperfectly because I hardcode the function $w()$ in the `ode()` function. It's just a shape issue. I am currently solving it over the time interval $[0, 1]$, and then solving for $t_f$ at the same time I solve for the optimal route. I pass that in as the argument `p`, and rescale everything in the `ode()` and `bc()` functions accordingly. I will include more here about the change of variables eventually.

# Derivation of Zermelo with variable wind

## Set up

$$J[u] = \int_0^{t_f} 1 dt \\ 
\text{ subj. to } \begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = v(\theta(t), \phi(x, y)) \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} - \begin{bmatrix} w_1(x, y) \\ w_2(x, y) \end{bmatrix} \\ 
\begin{bmatrix} x(0) \\ y(0) \end{bmatrix} = \begin{bmatrix} x_0 \\ y_0 \end{bmatrix} \\ \\
\begin{bmatrix} x(t_f) \\ y(t_f) \end{bmatrix} = \begin{bmatrix} x_f \\ y_f \end{bmatrix}$$

Where $\phi(x, y)$ gives the $x$ and $y$ components of the velocity of the wind.
So then the hamiltonian is given by 

$$ 
\begin{aligned}
H &= \lambda(t)^T (\dot{\vec{x}} - v(\theta(t), \phi(x, y)) \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} + \begin{bmatrix} w_1(x, y) \\ w_2(x, y) \end{bmatrix}) - 1\\
&= \begin{bmatrix} \lambda_1 & \lambda_2 \end{bmatrix}  \begin{bmatrix} \dot{x} - v(\theta(t), \phi(x, y)) \cos(\theta) - w_1(x, y) \\ \dot{y} - v(\theta(t), \phi(x, y)) \sin(\theta) - w_2(x, y) \end{bmatrix} - 1 \\

&= \lambda_1​(v(\theta, \phi(x, y))\cos(\theta) - w_1​)+ \lambda_2​(v(\theta,\phi(x, y))\sin(\theta)-w_2​)−1
\end{aligned}
$$

So the costate evolution equations are given by 

$$
\begin{aligned}
\dot{\lambda_1} &= -\frac{\partial H}{\partial x} \\
& = - \left ( \lambda_1 \left ( \frac{\partial V}{\partial x} \cos(\theta) + \frac{\partial w_1}{\partial x} \right ) - \lambda_2 \left ( \frac{\partial V}{\partial x} \sin(\theta) + \frac{\partial w_2}{\partial x} \right ) \right ) \\

\dot{\lambda_2} & = -\frac{\partial H}{\partial y} \\
& = - \left ( \lambda_1 \left ( \frac{\partial V}{\partial y} \cos(\theta) + \frac{\partial w_1}{\partial y} \right ) - \lambda_2 \left ( \frac{\partial V}{\partial y} \sin(\theta) + \frac{\partial w_2}{\partial y} \right ) \right ) \\

\frac{\partial H}{\partial \theta} &= \lambda_1 \left (\frac{\partial v}{\partial \theta} \cos(\theta) - v \sin(\theta) \right ) + \lambda_2 \left (\frac{\partial v}{\partial \theta} \sin(\theta) + v \cos(\theta) \right ) \\

&= 0

\end{aligned}
$$

And it is all subject to the boundary conditions
$$ \vec{s_{0}} = \vec{s}(0) \text{ and } \vec{s_{t_f}} = \vec{s}(t_f) \text{ and } H(t_f) = 0 $$

## Implementation
The implementation will be kinda tricky and maybe slow--if the sailing polar diagram is not well behaved, as it shouldn't be, then we will do a little optimization step inside of the ode() function to get the optimal $u^*$ at each step. As in

```python
def ode(t, Y):
  x, y, lam1, lam2 = Y
  theta_star = scipy.optimize.minimize(-H, args=(theta, x, y, lam1, lam2))

  dx, dy = ...
```
