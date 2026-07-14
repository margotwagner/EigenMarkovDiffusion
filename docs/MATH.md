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

## Discrete-time reference

The stochastic implementations use a finite transition step

\[
P=I-\Delta t A,
\qquad
\mathbf n_{s+1}=P\mathbf n_s.
\]

Therefore the exact expected node counts after integer step \(s\) are

\[
\mathbb E[\mathbf n_s]=P^s\mathbf n_0.
\]

In the eigenbasis, this becomes

\[
\mathbb E[m_k(s)]
=(1-\lambda_k\Delta t)^s m_k(0).
\]

This is distinct from the continuous solution
\(e^{-\lambda_k s\Delta t}m_k(0)\). Their difference is a timestep effect,
not necessarily a model failure.

## Exact multinomial statistics

When \(N\) independent particles begin at the same node, each particle has node
probability vector \(\mathbf p_s\) after step \(s\). The node-count vector is
marginally

\[
\mathbf n_s\sim\operatorname{Multinomial}(N,\mathbf p_s).
\]

Its node variances are

\[
\operatorname{Var}[n_i(s)]
=Np_{s,i}(1-p_{s,i}),
\]

and its same-time spatial covariance is

\[
\operatorname{Cov}(\mathbf n_s)
=N\left[\operatorname{diag}(\mathbf p_s)
-\mathbf p_s\mathbf p_s^\top\right].
\]

Transforming to modal coordinates gives

\[
\operatorname{Cov}(\mathbf m_s)
=V^\top\operatorname{Cov}(\mathbf n_s)V.
\]

This modal covariance is generally not diagonal. Consequently, evolving every
mode as an independent stochastic process may reproduce the modal means while
failing to reproduce the correct spatial variance and covariance.
