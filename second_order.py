import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# second order inertial element
w_n = 10
kesi = 0.707
s = ctrl.tf('s')
sys = w_n*w_n / (s*s + 2*kesi*w_n*s + w_n*w_n)

T = []
X0 = 0

cycle_num = 100
for i in np.arange(1,cycle_num):
    T.append(i*0.01)

t, response = ctrl.impulse_response(sys, T, X0)


# plot
plt.figure()
plt.title('second_order impulse response')
plt.plot(t, response,label='impulse response')
plt.legend()
plt.grid()
plt.show()