import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import curve_fit

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

# output noise
target_snr_db = 20
response_watts = response ** 2
sig_avg_watts = np.mean(response_watts)
sig_avg_db = 10 * np.log10(sig_avg_watts)
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
mean_noise = 0
noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(response_watts))

y_out = response + noise

# ID: fit for a params
# Time domin | Laplace Transform | Laplace Domain
# wn/sqrt(1-kesi*kesi)*exp(-kesi*wn*t)*sin(wn*sqrt(1-kesi*kesi)*t) 
# ----- Laplace Transform -----> 
#w_n*w_n / (s*s + 2*kesi*w_n*s + w_n*w_n)

def func(x, wn, kesi,k):
    sqrt_kesi = np.sqrt(1-kesi*kesi)
    return k*wn/sqrt_kesi*np.exp(-kesi*wn*x)*np.sin(wn*sqrt_kesi*x)

popt, pcov = curve_fit(func, t, y_out)
print("popt: ",popt)
print("pcov: ",pcov)

sys_fit = popt[0] / (s+popt[1])
t_fit, response_fit = ctrl.impulse_response(sys_fit, T, X0)


# plot
plt.figure()
plt.title('second_order impulse response')
plt.plot(t, response,label='without noise')
plt.plot(t, y_out,label='with noise')
plt.plot(t_fit, response_fit,label='fit')
plt.legend()
plt.grid()
plt.show()