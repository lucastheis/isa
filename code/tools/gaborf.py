
def gaborf

patch_size = sqrt(self.num_visibles)
values = linspace(-5, 5, patch_size)
xx, yy = meshgrid(values, values)

t = rand() * pi * 2.
a = rand() / 2. + 0.5
s = rand() * 1. + 1.
f = rand() * 2. + 2.

m, n = rand() * 10. - 5., rand() * 10. - 5.

x, y = xx - m, yy - n
x, y = x * cos(t) + y * sin(t), -x * sin(t) + y * cos(t)

G = exp(-square(x) / s - square(a * y) / s) * sin(f * x)
