# ISA

A Python implementation of overcomplete independent subspace analysis.

This code implements an efficient blocked Gibbs sampler for inference and maximum likelihood
learning in overcomplete linear models with sparse source distributions. A faster and more memory
efficient implementation written in C++ can be found here:

[https://github.com/lucastheis/cisa](https://github.com/lucastheis/cisa)

## Requirements

* Python >= 2.6.0
* NumPy >= 1.6.2
* SciPy >= 0.11.0

I have tested the code with the above versions, but older versions might also work.

## Reference

L. Theis, J. Sohl-Dickstein, and M. Bethge, *Training sparse natural image models with a fast Gibbs
sampler of an extended state space*, Advances in Neural Information Processing Systems 25, 2012
