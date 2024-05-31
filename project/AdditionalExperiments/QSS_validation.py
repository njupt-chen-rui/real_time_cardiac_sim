import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
import argparse
import os


@ti.data_oriented
class aliev_panfilov:
    def __init__(self, Vm_init, w_init, Cm, k, a, epsilon0, mu1, mu2, b):
        self.Vm = Vm_init
        self.w = w_init
        self.Cm = Cm
        self.k = k
        self.a = a
        self.epsilon0 = epsilon0
        self.mu1 = mu1
        self.mu2 = mu2
        self.b = b

    def forward_euler(self, dt):
        Vm_ = -1.0 * dt / self.Cm * (
                    self.k * self.Vm * (self.Vm - self.a) * (self.Vm - 1.0) + self.w * self.Vm) + self.Vm
        epsilon_ = self.epsilon0 + self.mu1 * self.w / (self.mu2 + self.Vm)
        w_ = epsilon_ * dt * (-1.0 * self.k * self.Vm * (self.Vm - self.b - 1.0) - self.w) + self.w
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
        self.Vm = self.Vm * tm.exp(-1.0 * (self.k * self.a + self.w) / self.Cm * dt) + (
                (self.k * self.Vm * self.Vm * (1.0 + self.a - self.Vm)) / (self.k * self.a + self.w)) * (
                          1.0 - tm.exp(-1.0 * (self.k * self.a + self.w) / self.Cm * dt))

    def Rw(self, dt):
        epsilon_ = self.epsilon0 + self.mu1 * self.w / (self.mu2 + self.Vm)
        self.w = self.w * tm.exp(-1.0 * epsilon_ * dt) + self.k * self.Vm * (1.0 + self.b - self.Vm) * (
                    1.0 - tm.exp(-1.0 * epsilon_ * dt))

    def Rz(self, dt):
        pass

    # def backward_euler(self, dt):
    #     p1 = dt / self.Cm
    #     p2 = -1.0 * dt * (self.a + 1.0) / self.Cm
    #     p3 = 1.0 + self.a * dt / self.Cm + self.epsilon0 * self.beta * dt * dt / (
    #                 self.Cm * (1.0 + self.epsilon0 * self.gamma * dt))
    #     p4 = dt * (self.w - self.epsilon0 * self.sigma * dt) / (
    #                 self.Cm * (1.0 + self.epsilon0 * self.gamma * dt)) - self.Vm
    #
    #     # 牛顿迭代
    #     delta = 0.001
    #     while 1:
    #         fx_0 = p1 * self.Vm * self.Vm * self.Vm + p2 * self.Vm * self.Vm + p3 * self.Vm + p4
    #         dfx_0 = 3.0 * p1 * self.Vm * self.Vm + 2.0 * p2 * self.Vm + p3
    #         Vm_ = self.Vm - fx_0 / dfx_0
    #         fx_new = p1 * Vm_ * Vm_ * Vm_ + p2 * Vm_ * Vm_ + p3 * Vm_ + p4
    #         print(abs(fx_new - fx_0))
    #         old_diff = abs(fx_new - fx_0)
    #         if abs(fx_new - fx_0) < delta:
    #             break
    #         if abs(abs(fx_new - fx_0) - old_diff) < 0.0000001:
    #             break
    #
    #     self.Vm = Vm_
    #
    #     self.w = (self.epsilon0 * self.beta * dt * self.Vm + self.w - self.epsilon0 * self.sigma * dt) / (
    #                 1.0 + self.epsilon0 * self.gamma * dt)


@ti.data_oriented
class fitzhugh_nagumo:
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
        # self.Vm = self.Vm * tm.exp(-1.0 * self.a / self.Cm * dt) + (
        #             (self.Vm * ((1.0 + self.a) * self.Vm - self.Vm * self.Vm)) - self.w / self.a) * (
        #                   1.0 - tm.exp(-1.0 * self.a / self.Cm * dt))
        self.Vm = self.Vm * tm.exp(-1.0 * self.a / self.Cm * dt) + (
                (self.Vm * ((1.0 + self.a) * self.Vm - self.Vm * self.Vm) - self.w) / self.a) * (
                          1.0 - tm.exp(-1.0 * self.a / self.Cm * dt))

    def Rw(self, dt):
        # self.w = self.w * tm.exp(-1.0 * self.gamma * self.epsilon0 * dt) + (
        #         self.beta * self.Vm * (self.Vm - self.sigma)) / self.gamma * (
        #                  1.0 - tm.exp(-1.0 * self.gamma * self.epsilon0 * dt))
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
            # print(abs(fx_new - fx_0))
            old_diff = abs(fx_new - fx_0)
            if abs(fx_new - fx_0) < delta:
                break
            if abs(abs(fx_new - fx_0) - old_diff) < 0.0000001:
                break

        self.Vm = Vm_

        self.w = (self.epsilon0 * self.beta * dt * self.Vm + self.w - self.epsilon0 * self.sigma * dt) / (
                    1.0 + self.epsilon0 * self.gamma * dt)


def elec_reaction_fn(args):
    steps = int(args.T / args.dt)
    dt = args.dt
    fn1 = fitzhugh_nagumo(Vm_init=args.Vm0, w_init=args.w0, Cm=args.Cm, a=args.a, epsilon0=args.ep0, beta=args.beta, gamma=args.gamma, sigma=args.sigma)
    fn2 = fitzhugh_nagumo(Vm_init=args.Vm0, w_init=args.w0, Cm=args.Cm, a=args.a, epsilon0=args.ep0, beta=args.beta, gamma=args.gamma, sigma=args.sigma)
    fn3 = fitzhugh_nagumo(Vm_init=args.Vm0, w_init=args.w0, Cm=args.Cm, a=args.a, epsilon0=args.ep0, beta=args.beta, gamma=args.gamma, sigma=args.sigma)
    fn1_x = np.zeros(steps)
    fn1_y = np.zeros(steps)
    fn2_x = np.zeros(steps)
    fn2_y = np.zeros(steps)
    fn3_x = np.zeros(steps)
    fn3_y = np.zeros(steps)

    for i in range(steps):
        fn1.forward_euler(dt)
        fn1_x[i] = i * dt
        fn1_y[i] = fn1.Vm

        fn2.QSS(dt)
        fn2_x[i] = i * dt
        fn2_y[i] = fn2.Vm

        fn3.backward_euler(dt)
        fn3_x[i] = i * dt
        fn3_y[i] = fn3.Vm

    plt.plot(fn1_x, fn1_y, ls='-.', label='前向欧拉法')
    plt.plot(fn2_x, fn2_y, ls='--', label='QSS方法')
    plt.plot(fn3_x, fn3_y, ls=':', label='后向欧拉法')
    plt.title(r"FitzHugh Nagumo模型，$\Delta t=$" + str(dt), fontsize=17)
    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$V_{m}$", fontsize=15)
    plt.legend(fontsize=15)

    # 保存图片
    path = args.save_path + 'QSS_validation_fn_' + str(dt) + '.svg'
    plt.savefig(path, dpi=1200, format='svg')

    plt.show()


def elec_reaction_ap(args):
    dt = args.dt
    steps = int(args.T / args.dt)
    ap1 = aliev_panfilov(Vm_init=args.Vm0, w_init=args.w0, Cm=args.Cm, k=args.k, a=args.a, epsilon0=args.ep0, mu1=args.mu1, mu2=args.mu2, b=args.b)
    ap2 = aliev_panfilov(Vm_init=args.Vm0, w_init=args.w0, Cm=args.Cm, k=args.k, a=args.a, epsilon0=args.ep0, mu1=args.mu1, mu2=args.mu2, b=args.b)
    ap1_x = np.zeros(steps)
    ap1_y = np.zeros(steps)
    ap2_x = np.zeros(steps)
    ap2_y = np.zeros(steps)

    for i in range(steps):
        ap1.forward_euler(dt)
        ap1_x[i] = i * dt
        ap1_y[i] = ap1.Vm

        ap2.QSS(dt)
        ap2_x[i] = i * dt
        ap2_y[i] = ap2.Vm

    plt.plot(ap1_x, ap1_y, ls='-.', label='前向欧拉法')
    plt.plot(ap2_x, ap2_y, ls='--', label='QSS方法')
    plt.title(r"Aliev Panfilov模型，$\Delta t=$" + str(dt), fontsize=17)
    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$V_{m}$", fontsize=15)
    plt.legend(fontsize=15)

    # 保存图片
    path = args.save_path + 'QSS_validation_ap_' + str(dt) + '.svg'
    plt.savefig(path, dpi=1200, format='svg')
    
    plt.show()


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'STFangsong'  # 替换为你选择的字体
    ti.init(arch=ti.cuda, default_fp=ti.f32)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./res/')
    parser.add_argument('--model', type=str, default='fn')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--T', type=float, default=400.0)
    parser.add_argument('--Vm0', type=float, default=1.0)
    parser.add_argument('--w0', type=float, default=0.1)
    parser.add_argument('--Cm', type=float, default=1.0)
    parser.add_argument('--k', type=float, default=0.5)
    parser.add_argument('--a', type=float, default=0.1)
    parser.add_argument('--ep0', type=float, default=0.01)
    parser.add_argument('--mu1', type=float, default=0.2)
    parser.add_argument('--mu2', type=float, default=0.3)
    parser.add_argument('--b', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=0.0)

    # --model=ap --Vm0=0.5 --w0=0.0 --a=0.15 --ep0=0.02

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model == 'fn':
        elec_reaction_fn(args)
    elif args.model == 'ap':
        elec_reaction_ap(args)
