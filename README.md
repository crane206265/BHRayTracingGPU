# BHRayTracing
**: Black hole randering code by ray tracing**

## Preview
#### [Example 1]
**Settings**
- Schwarzchild Black Hole
- Mass of BH : $M=1.00$
- inclination : $i=80\degree$
<br><br>
![BH_Example1](/demo_version/saved_data/BH_img.png)

-----
## File Tree
```
src:.
│  camera.py
│  events.py
│  hamiltonian.py
│  types.py
│
├─cache
│      reader.py
│      writer.py
│
├─integrators
│      batched.py
│      rk4.py
│
├─metrics
│      base.py
│      kerr_schild.py
│      minkowski.py
│
└─plasma
        base.py
        disk.py
```
- `metrics/` : metric calculation
- `camera.py` : camera setting / tetrad calculation
- `hamiltonian.py` : geodesic RHS
- `integrators/rk4.py` : RK4
- `integrators/batched.py` : GPU ray batching
- `events.py` : ray termination
- `cache/` : history save / read
- `plasma/` : disk emission

**\[Dependence\]**
: metric -> Hamiltonian -> RK4 -> batched integrator -> ray history / cache



-----
## Basic
### Geodesic
Geodesic Equation:

$$\frac{d^2x^\mu}{d\lambda^2}+{\Gamma^{\mu}}_{\alpha\beta}\frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda}=0$$

or

$$U^\alpha\nabla_\alpha U^\mu=0$$

### Schwarzchild Black Hole
Metric:

$$ds^2=-\left(1-\frac{r_s}{r}\right)dt^2+\left(1-\frac{r_s}{r}\right)^{-1}dr^2+r^2d\theta^2+r^2\sin^2\theta d\phi^2$$

where $r_s=\frac{2GM}{c^2}$ is Schwarzchild radius.

