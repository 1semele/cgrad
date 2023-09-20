# cgrad

a simple from-scratch implementation of an autodiff engine.

i first implemented it for scalar values to learn the core idea, and
then i implemented it for tensors.

concerns of speed/memory usage/convenience are secondary to getting the gradient stuff right.

may be of interest for those learning.

# building

requires a c compiler and GNU make.

```
make
```

# resources i used

[reverse mode autodiff tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
[micrograd](https://github.com/karpathy/micrograd)
[numpy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
