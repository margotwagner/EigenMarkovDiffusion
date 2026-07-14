# Mathematical formulation

The one-dimensional diffusion equation is

\[
\frac{\partial u(x,t)}{\partial t}=D\frac{\partial^2u(x,t)}{\partial x^2}.
\]

After discretizing space, the particle-count vector \(\mathbf n(t)\) obeys

\[
\frac{d\mathbf n}{dt}=-A\mathbf n,
\]

where \(A\) is the positive-semidefinite Neumann graph Laplacian scaled by
\(k=D/\Delta x^2\). Its endpoint diagonal entries are \(k\), its interior
entries are \(2k\), and its neighbouring off-diagonal entries are \(-k\).

Because \(A\) is symmetric,

\[
A=V\Lambda V^\top,
\qquad
\mathbf m=V^\top\mathbf n.
\]

The deterministic modes evolve independently:

\[
\frac{dm_k}{dt}=-\lambda_km_k,
\qquad
m_k(t)=e^{-\lambda_kt}m_k(0).
\]

EigenMarkov represents each nonconstant modal amplitude as the difference of
two state populations,

\[
m_k=w(q_k^+-q_k^-),
\]

where \(w\) is the modal-particle weight. During a timestep \(\Delta t\), each
state unit switches sign with probability

\[
p_k=\frac{\lambda_k\Delta t}{2}.
\]

Therefore,

\[
\mathbb E[m_k(t+\Delta t)]
=(1-2p_k)\mathbb E[m_k(t)]
=(1-\lambda_k\Delta t)\mathbb E[m_k(t)],
\]

which is the first-order diffusion update. The zero mode has \(\lambda_0=0\)
and preserves total mass. Spatial values are reconstructed with

\[
\mathbf n=V\mathbf m.
\]

Retaining only the first \(M\) slow modes gives the reduced approximation

\[
\mathbf n(t)\approx V_M\mathbf m_M(t),\qquad M<N.
\]

The signed reconstruction is not guaranteed to be nonnegative. This is an
explicitly measured limitation rather than something silently repaired by the
implementation.
