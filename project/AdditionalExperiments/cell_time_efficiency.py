import time
import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


@ti.data_oriented
class fitzhugh_nagumo_CPU:
    def __init__(self, Vm_init, w_init, Cm, a, epsilon0, beta, gamma, sigma):
        self.Vm = Vm_init
        self.w = w_init
        self.Cm = Cm
        self.a = a
        self.epsilon0 = epsilon0
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward_euler(self, dt):
        Vm_ = -dt / self.Cm * (self.Vm * (self.Vm - self.a) * (self.Vm - 1.0) + self.w) + self.Vm
        w_ = self.epsilon0 * dt * (self.beta * self.Vm - self.gamma * self.w - self.sigma) + self.w
        self.Vm = Vm_
        self.w = w_

    def QSS(self, dt):
        self.Rv(dt * 0.5)
        self.Rw(dt * 0.5)
        self.Rz(dt * 0.5)
        self.Rz(dt * 0.5)
        self.Rw(dt * 0.5)
        self.Rv(dt * 0.5)

    def Rv(self, dt):
        self.Vm = self.Vm * tm.exp(-1.0 * self.a / self.Cm * dt) + (
                (self.Vm * ((1.0 + self.a) * self.Vm - self.Vm * self.Vm) - self.w) / self.a) * (
                          1.0 - tm.exp(-1.0 * self.a / self.Cm * dt))

    def Rw(self, dt):
        self.w = self.w * tm.exp(-1.0 * self.gamma * self.epsilon0 * dt) + (
                    self.beta * self.Vm - self.sigma) / self.gamma * (
                         1.0 - tm.exp(-1.0 * self.gamma * self.epsilon0 * dt))

    def Rz(self, dt):
        pass

    def backward_euler(self, dt):
        p1 = dt / self.Cm
        p2 = -1.0 * dt * (self.a + 1.0) / self.Cm
        p3 = 1.0 + self.a * dt / self.Cm + self.epsilon0 * self.beta * dt * dt / (
                    self.Cm * (1.0 + self.epsilon0 * self.gamma * dt))
        p4 = dt * (self.w - self.epsilon0 * self.sigma * dt) / (
                    self.Cm * (1.0 + self.epsilon0 * self.gamma * dt)) - self.Vm

        # 牛顿迭代
        delta = 0.001
        while 1:
            fx_0 = p1 * self.Vm * self.Vm * self.Vm + p2 * self.Vm * self.Vm + p3 * self.Vm + p4
            dfx_0 = 3.0 * p1 * self.Vm * self.Vm + 2.0 * p2 * self.Vm + p3
            Vm_ = self.Vm - fx_0 / dfx_0
            fx_new = p1 * Vm_ * Vm_ * Vm_ + p2 * Vm_ * Vm_ + p3 * Vm_ + p4
            print(abs(fx_new - fx_0))
            old_diff = abs(fx_new - fx_0)
            if abs(fx_new - fx_0) < delta:
                break
            if abs(abs(fx_new - fx_0) - old_diff) < 0.0000001:
                break

        self.Vm = Vm_

        self.w = (self.epsilon0 * self.beta * dt * self.Vm + self.w - self.epsilon0 * self.sigma * dt) / (
                    1.0 + self.epsilon0 * self.gamma * dt)


@ti.data_oriented
class fitzhugh_nagumo_GPU:
    def __init__(self, Cm, a, epsilon0, beta, gamma, sigma, scale):
        self.n = scale
        self.Vm = ti.field(float, shape=(scale,))
        self.w = ti.field(float, shape=(scale,))
        self.Cm = Cm
        self.a = a
        self.epsilon0 = epsilon0
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    @ti.kernel
    def init_Vm_w(self):
        for i in self.Vm:
            self.Vm[i] = 1.0 / self.n * i
            self.w[i] = 0.1

    def QSS(self, dt):
        self.Rv(dt * 0.5)
        self.Rw(dt * 0.5)
        self.Rz(dt * 0.5)
        self.Rz(dt * 0.5)
        self.Rw(dt * 0.5)
        self.Rv(dt * 0.5)

    @ti.kernel
    def Rv(self, dt: float):
        for i in self.Vm:
            self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * self.a / self.Cm * dt) + (
                    (self.Vm[i] * ((1.0 + self.a) * self.Vm[i] - self.Vm[i] * self.Vm[i]) - self.w[i]) / self.a) * (
                              1.0 - tm.exp(-1.0 * self.a / self.Cm * dt))

    @ti.kernel
    def Rw(self, dt: float):
        for i in self.w:
            self.w[i] = self.w[i] * tm.exp(-1.0 * self.gamma * self.epsilon0 * dt) + (
                    self.beta * self.Vm[i] - self.sigma) / self.gamma * (
                             1.0 - tm.exp(-1.0 * self.gamma * self.epsilon0 * dt))

    def Rz(self, dt):
        pass


def elec_reaction_fn_time_efficiency_CPU(dt, steps, grid_scale):
    fn_cpu_array = []
    for i in range(grid_scale):
        fn_cpu_array.append(
            fitzhugh_nagumo_CPU(Vm_init=1.0 / grid_scale * i, w_init=0.1, Cm=1.0, a=0.1, epsilon0=0.01, beta=0.5,
                                gamma=1.0, sigma=0.0))

    T1 = time.perf_counter()
    for j in range(steps):
        for i in range(grid_scale):
            fn_cpu_array[i].QSS(dt)
    T2 = time.perf_counter()
    print('网格规模:' + str(grid_scale))
    print('串行算法计算用时: %s 毫秒' % ((T2 - T1) * 1000))


def elec_reaction_fn_time_efficiency_GPU(dt, steps, grid_scale):
    fn_gpu_array = fitzhugh_nagumo_GPU(Cm=1.0, a=0.1, epsilon0=0.01, beta=0.5, gamma=1.0, sigma=0.0, scale=grid_scale)
    fn_gpu_array.init_Vm_w()

    # print(fn_gpu_array.Vm[50])
    T1 = time.perf_counter()
    for j in range(steps):
        fn_gpu_array.QSS(dt)

    # print(fn_gpu_array.Vm[50])
    T2 = time.perf_counter()
    print('网格规模:' + str(grid_scale))
    print('并行算法计算用时: %s 毫秒' % ((T2 - T1) * 1000))


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'STFangsong'  # 替换为你选择的字体
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    elec_reaction_fn_time_efficiency_CPU(dt=0.1, steps=1000, grid_scale=100)
    elec_reaction_fn_time_efficiency_CPU(dt=0.1, steps=1000, grid_scale=1000)
    elec_reaction_fn_time_efficiency_CPU(dt=0.1, steps=1000, grid_scale=10000)
    # elec_reaction_fn_time_efficiency_CPU(dt=0.1, steps=1000, grid_scale=100000)

    elec_reaction_fn_time_efficiency_GPU(dt=0.1, steps=1000, grid_scale=100)
    elec_reaction_fn_time_efficiency_GPU(dt=0.1, steps=1000, grid_scale=1000)
    elec_reaction_fn_time_efficiency_GPU(dt=0.1, steps=1000, grid_scale=10000)
    elec_reaction_fn_time_efficiency_GPU(dt=0.1, steps=1000, grid_scale=100000)
