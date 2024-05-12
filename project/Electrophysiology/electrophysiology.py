import taichi as ti
import taichi.math as tm
from project.Geometry import Body


@ti.data_oriented
class Electrophysiology:
    """用于仿真心肌组织的电活动的基类，实现了扩散项的计算以及反应项的分解，具体反应项模型由子类控制

    属性:
        Vm: 顶点电压
        w: 门控变量
        I_ext: 外界电刺激
        sigma_f, sigma_s, sigma_n: 扩散项各向异性权重
        cg_epsilon: 共轭梯度法容差
    """

    def __init__(self, body: Body, sigma_f=1.1, sigma_s=1.0, sigma_n=1.0):
        """初始化Electrophysiology类

        :param body: 物体的几何属性
        """

        # 几何数据
        self.body = body

        # 电压和门控变量
        self.Vm = self.body.Vm
        self.w = ti.field(float, shape=body.num_nodes)
        # TODO: 外界电刺激的施加
        self.I_ext = ti.field(float, shape=body.num_nodes)
        self.init_elec()

        # 扩散项各向异性权重
        self.sigma_f = sigma_f
        self.sigma_s = sigma_s
        self.sigma_n = sigma_n

        # 描述形变影响的辅助数据
        self.Dm = ti.Matrix.field(3, 3, float, shape=body.num_elements)
        self.DmInv = ti.Matrix.field(3, 3, float, shape=body.num_elements)
        self.Ds = ti.Matrix.field(3, 3, float, shape=body.num_elements)
        self.F = ti.Matrix.field(3, 3, float, shape=body.num_elements)
        self.Be = ti.Matrix.field(3, 3, float, shape=body.num_elements)
        self.init_Ds_F_Be()
        self.fiber = ti.Vector.field(3, float, shape=body.num_elements)
        self.sheet = ti.Vector.field(3, float, shape=body.num_elements)
        self.normal = ti.Vector.field(3, float, shape=body.num_elements)
        self.init_fiber()
        self.DM = ti.Matrix.field(3, 3, float, shape=body.num_elements)
        self.Me = ti.Matrix.field(4, 4, float, shape=body.num_elements)
        self.Ke = ti.Matrix.field(4, 4, float, shape=body.num_elements)
        self.D = ti.Matrix.field(3, 3, float, shape=body.num_elements)

        # 共轭梯度法辅助变量
        self.cg_x = ti.field(float, self.body.num_nodes)
        self.cg_Ax = ti.field(float, self.body.num_nodes)
        self.cg_b = ti.field(float, self.body.num_nodes)
        self.cg_r = ti.field(float, self.body.num_nodes)
        self.cg_d = ti.field(float, self.body.num_nodes)
        self.cg_Ad = ti.field(float, self.body.num_nodes)
        self.cg_epsilon = 1.0e-3

    @ti.kernel
    def init_elec(self):
        """初始化电压，门控变量和外界刺激

        :return:
        """
        for i in self.Vm:
            self.Vm[i] = 0.0
            self.w[i] = 0.0
            self.I_ext[i] = 0.0

    @ti.kernel
    def init_Ds_F_Be(self):
        """初始化形变梯度以及有关变量

        :return:
        """

        # 语法糖
        nodes = ti.static(self.body.nodes)
        elements = ti.static(self.body.elements)

        # 初始化Dm以及其逆矩阵
        for i in elements:
            self.Dm[i][0, 0] = nodes[elements[i][0]][0] - nodes[elements[i][3]][0]
            self.Dm[i][1, 0] = nodes[elements[i][0]][1] - nodes[elements[i][3]][1]
            self.Dm[i][2, 0] = nodes[elements[i][0]][2] - nodes[elements[i][3]][2]
            self.Dm[i][0, 1] = nodes[elements[i][1]][0] - nodes[elements[i][3]][0]
            self.Dm[i][1, 1] = nodes[elements[i][1]][1] - nodes[elements[i][3]][1]
            self.Dm[i][2, 1] = nodes[elements[i][1]][2] - nodes[elements[i][3]][2]
            self.Dm[i][0, 2] = nodes[elements[i][2]][0] - nodes[elements[i][3]][0]
            self.Dm[i][1, 2] = nodes[elements[i][2]][1] - nodes[elements[i][3]][1]
            self.Dm[i][2, 2] = nodes[elements[i][2]][2] - nodes[elements[i][3]][2]
        for i in self.DmInv:
            self.DmInv[i] = self.Dm[i].inverse()

        # 初始化Ds矩阵
        for i in self.Ds:
            self.Ds[i][0, 0] = nodes[elements[i][0]][0] - nodes[elements[i][3]][0]
            self.Ds[i][1, 0] = nodes[elements[i][0]][1] - nodes[elements[i][3]][1]
            self.Ds[i][2, 0] = nodes[elements[i][0]][2] - nodes[elements[i][3]][2]
            self.Ds[i][0, 1] = nodes[elements[i][1]][0] - nodes[elements[i][3]][0]
            self.Ds[i][1, 1] = nodes[elements[i][1]][1] - nodes[elements[i][3]][1]
            self.Ds[i][2, 1] = nodes[elements[i][1]][2] - nodes[elements[i][3]][2]
            self.Ds[i][0, 2] = nodes[elements[i][2]][0] - nodes[elements[i][3]][0]
            self.Ds[i][1, 2] = nodes[elements[i][2]][1] - nodes[elements[i][3]][1]
            self.Ds[i][2, 2] = nodes[elements[i][2]][2] - nodes[elements[i][3]][2]

        # 初始化Be矩阵
        # TODO: 这里的 Be 矩阵和 Body 类中的 Dm 矩阵一样，可以考虑统一
        for i in self.Be:
            self.Be[i][0, 0] = nodes[elements[i][1]][0] - nodes[elements[i][0]][0]
            self.Be[i][1, 0] = nodes[elements[i][1]][1] - nodes[elements[i][0]][1]
            self.Be[i][2, 0] = nodes[elements[i][1]][2] - nodes[elements[i][0]][2]
            self.Be[i][0, 1] = nodes[elements[i][2]][0] - nodes[elements[i][0]][0]
            self.Be[i][1, 1] = nodes[elements[i][2]][1] - nodes[elements[i][0]][1]
            self.Be[i][2, 1] = nodes[elements[i][2]][2] - nodes[elements[i][0]][2]
            self.Be[i][0, 2] = nodes[elements[i][3]][0] - nodes[elements[i][0]][0]
            self.Be[i][1, 2] = nodes[elements[i][3]][1] - nodes[elements[i][0]][1]
            self.Be[i][2, 2] = nodes[elements[i][3]][2] - nodes[elements[i][0]][2]

        # 初始化形变梯度 F 矩阵
        for i in self.F:
            self.F[i] = self.Ds[i] @ self.DmInv[i]

    @ti.kernel
    def init_fiber(self):
        """初始化纤维方向(网格上)，考虑了初始形变的影响

        :return:
        """
        for i in range(self.body.num_elements):
            self.fiber[i] = self.F[i] @ self.body.tet_fiber[i]
            self.sheet[i] = self.F[i] @ self.body.tet_sheet[i]
            self.normal[i] = self.F[i] @ tm.cross(self.body.tet_fiber[i], self.body.tet_sheet[i])

    def set_para_diffusion(self, sigma_f=1.1, sigma_s=1.0, sigma_n=1.0):
        """设置电活动中扩散项的各向异性权重

        :param sigma_f: fiber方向的权重
        :param sigma_s: sheet方向的权重
        :param sigma_n: normal方向的权重
        :return:
        """

        self.sigma_f = sigma_f
        self.sigma_s = sigma_s
        self.sigma_n = sigma_n

    def set_para_cg_epsilon(self, cg_epsilon=1.0e-3):
        """设置共轭梯度法的计算误差cg_epsilon, 默认值为0.001

        :param cg_epsilon: 设置的计算误差
        :return:
        """

        self.cg_epsilon = cg_epsilon

    def update(self, sub_steps):
        """时间步进更新电活动

        :param sub_steps: 子步数量
        :return:
        """
        dt = 1. / 1.29 / 6. / sub_steps
        for _ in range(sub_steps):
            self.update_Vm(dt)

    # TODO: 在doc中添加多尺度电生理系统解耦的说明
    def update_Vm(self, dt):
        """使用2阶Strang算子分裂将多尺度的电活动解耦为:
            - 细胞级的反应项
            - 组织级的扩散项

        :param dt: 时间步长
        :return:
        """

        self.calculate_reaction(dt * 0.5)
        self.calculate_diffusion(dt)
        self.calculate_reaction(dt * 0.5)

    # TODO: 在doc中添加多反应项体系解耦为单反应项的说明
    @ti.kernel
    def calculate_reaction(self, dt: float):
        """使用 Reaction-by-Reaction splitting 解耦多反应系统
        TODO: 在具体电生理模型中添加Rz项的实现
        :param dt: 时间步长
        :return:
        """

        for i in self.Vm:
            self.calculate_Rv(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rz(i, dt * 0.5)
            self.calculate_Rz(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rv(i, dt * 0.5)

    # TODO: 在doc中添加使用有限元分析和共轭梯度法求解扩散项的说明
    def calculate_diffusion(self, dt):
        """使用共轭梯度法求解扩散项

        :param dt: 时间步长
        :return:
        """

        self.calculate_M_and_K()
        self.compute_RHS()
        self.cg(dt)
        self.cgUpdateVm()

    @ti.kernel
    def calculate_M_and_K(self):
        """计算单元中的局部质量矩阵M和刚度矩阵K

        :return:
        """

        fiber, sheet, normal = ti.static(self.fiber, self.sheet, self.normal)

        for i in range(self.body.num_elements):
            # 局部质量矩阵Me
            self.Me[i] = 0.25 / 6.0 * ti.abs(self.Be[i].determinant()) * \
                         tm.mat4([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]]) * 6.0

            J_phi = ti.Matrix([[-1., -1., -1.],
                               [1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]], float)
            D_f = ti.Matrix([[fiber[i][0] * fiber[i][0],
                              fiber[i][1] * fiber[i][0],
                              fiber[i][2] * fiber[i][0]],
                             [fiber[i][0] * fiber[i][1],
                              fiber[i][1] * fiber[i][1],
                              fiber[i][2] * fiber[i][1]],
                             [fiber[i][0] * fiber[i][2],
                              fiber[i][1] * fiber[i][2],
                              fiber[i][2] * fiber[i][2]]], float)
            D_s = ti.Matrix([[sheet[i][0] * sheet[i][0],
                              sheet[i][1] * sheet[i][0],
                              sheet[i][2] * sheet[i][0]],
                             [sheet[i][0] * sheet[i][1],
                              sheet[i][1] * sheet[i][1],
                              sheet[i][2] * sheet[i][1]],
                             [sheet[i][0] * sheet[i][2],
                              sheet[i][1] * sheet[i][2],
                              sheet[i][2] * sheet[i][2]]], float)
            D_n = ti.Matrix([[normal[i][0] * normal[i][0],
                              normal[i][1] * normal[i][0],
                              normal[i][2] * normal[i][0]],
                             [normal[i][0] * normal[i][1],
                              normal[i][1] * normal[i][1],
                              normal[i][2] * normal[i][1]],
                             [normal[i][0] * normal[i][2],
                              normal[i][1] * normal[i][2],
                              normal[i][2] * normal[i][2]]], float)
            norm_f = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            norm_s = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            norm_n = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            self.DM[i] = self.sigma_f / norm_f * D_f + self.sigma_s / norm_s * D_s + self.sigma_n / norm_n * D_n
            # 形变梯度的Jacobi
            J = self.F[i].determinant()
            A = J_phi @ self.Be[i].inverse() @ self.F[i].inverse()
            # 局部刚度矩阵Ke
            self.Ke[i] = 1.0 / 6.0 * J * ti.abs(self.Be[i].determinant()) * A @ self.DM[i] @ A.transpose() * 6.0
            # 电导率张量D
            self.D[i] = J * self.F[i].inverse() @ self.DM[i] @ self.F[i].inverse().transpose()

    @ti.kernel
    def compute_RHS(self):
        """计算右侧向量

        :return:
        """

        # 初始化为零
        for i in self.cg_b:
            self.cg_b[i] = 0.0

        # rhs = b = f * dt + M * u(t), here, f = 0
        for i in range(self.body.num_elements):
            # 四面体四个顶点的索引
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])

            self.cg_b[id0] += (self.Me[i][0, 0] * self.Vm[id0] + self.Me[i][0, 1] * self.Vm[id1] +
                               self.Me[i][0, 2] * self.Vm[id2] + self.Me[i][0, 3] * self.Vm[id3])
            self.cg_b[id1] += (self.Me[i][1, 0] * self.Vm[id0] + self.Me[i][1, 1] * self.Vm[id1] +
                               self.Me[i][1, 2] * self.Vm[id2] + self.Me[i][1, 3] * self.Vm[id3])
            self.cg_b[id2] += (self.Me[i][2, 0] * self.Vm[id0] + self.Me[i][2, 1] * self.Vm[id1] +
                               self.Me[i][2, 2] * self.Vm[id2] + self.Me[i][2, 3] * self.Vm[id3])
            self.cg_b[id3] += (self.Me[i][3, 0] * self.Vm[id0] + self.Me[i][3, 1] * self.Vm[id1] +
                               self.Me[i][3, 2] * self.Vm[id2] + self.Me[i][3, 3] * self.Vm[id3])

    def cg(self, dt: float):
        """共轭梯度法的计算

        :param dt: 时间步长
        :return:
        """

        delta_new = self.cg_before_ite(dt)
        delta_0 = delta_new

        # ite: 迭代次数, iteMax: 迭代次数最大值
        # TODO: 迭代次数最大值应该可调
        ite, iteMax = 0, 100
        while ite < iteMax and delta_new > (self.cg_epsilon**2) * delta_0:
            delta_new = self.cg_run_iteration(dt, delta_new)
            ite += 1

    @ti.kernel
    def cg_before_ite(self, dt: float) -> float:
        """共轭梯度法迭代前变量初始化

        :param dt: 时间步长
        :return: delta_new
        """

        for i in range(self.body.num_nodes):
            self.cg_x[i] = self.Vm[i]

        self.A_mult_x(dt, self.cg_Ax, self.cg_x)

        for i in range(self.body.num_nodes):
            # r = b - A @ x
            self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]
            # d = r
            self.cg_d[i] = self.cg_r[i]

        delta_new = self.dot(self.cg_r, self.cg_r)
        return delta_new

    @ti.func
    def A_mult_x(self, dt, dst, src):
        """并行化稀疏矩阵A乘自变量向量x

        :param dt: 时间步长
        :param dst: 目标向量Ax
        :param src: 源向量x
        :return:
        """

        # lhs = Ax = (M + K * dt) * u(t+1)
        for i in range(self.body.num_nodes):
            dst[i] = 0.0

        for i in range(self.body.num_elements):
            # id0~4为四面体中四个顶点的索引
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])

            # 计算目标向量Ax
            dst[id0] += (self.Me[i][0, 0] * src[id0] + self.Me[i][0, 1] * src[id1] +
                         self.Me[i][0, 2] * src[id2] + self.Me[i][0, 3] * src[id3])
            dst[id1] += (self.Me[i][1, 0] * src[id0] + self.Me[i][1, 1] * src[id1] +
                         self.Me[i][1, 2] * src[id2] + self.Me[i][1, 3] * src[id3])
            dst[id2] += (self.Me[i][2, 0] * src[id0] + self.Me[i][2, 1] * src[id1] +
                         self.Me[i][2, 2] * src[id2] + self.Me[i][2, 3] * src[id3])
            dst[id3] += (self.Me[i][3, 0] * src[id0] + self.Me[i][3, 1] * src[id1] +
                         self.Me[i][3, 2] * src[id2] + self.Me[i][3, 3] * src[id3])
            dst[id0] += (self.Ke[i][0, 0] * src[id0] + self.Ke[i][0, 1] * src[id1] +
                         self.Ke[i][0, 2] * src[id2] + self.Ke[i][0, 3] * src[id3]) * dt
            dst[id1] += (self.Ke[i][1, 0] * src[id0] + self.Ke[i][1, 1] * src[id1] +
                         self.Ke[i][1, 2] * src[id2] + self.Ke[i][1, 3] * src[id3]) * dt
            dst[id2] += (self.Ke[i][2, 0] * src[id0] + self.Ke[i][2, 1] * src[id1] +
                         self.Ke[i][2, 2] * src[id2] + self.Ke[i][2, 3] * src[id3]) * dt
            dst[id3] += (self.Ke[i][3, 0] * src[id0] + self.Ke[i][3, 1] * src[id1] +
                         self.Ke[i][3, 2] * src[id2] + self.Ke[i][3, 3] * src[id3]) * dt

    @ti.func
    def dot(self, v1, v2):
        """向量v1和向量v2的并行点积, v1和v2是高维向量时，加速效果明显

        :param v1: 高维向量v1
        :param v2: 高维向量v2
        :return: v1和v2的点积结果, 一个标量
        """

        result = 0.0
        for i in range(self.body.num_nodes):
            result += v1[i] * v2[i]
        return result

    @ti.kernel
    def cg_run_iteration(self, dt: float, delta: float) -> float:
        """共轭梯度法迭代过程

        :param dt: 时间步长
        :param delta: 共轭梯度法迭代中delta的初值
        :return: delta_new
        """

        delta_new = delta

        # q = A @ d
        self.A_mult_x(dt, self.cg_Ad, self.cg_d)

        # alpha = delta_new / d.dot(q)
        alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)

        for i in range(self.body.num_nodes):
            # x = x + alpha * d
            self.cg_x[i] += alpha * self.cg_d[i]
            # r = b - A @ x || r = r - alpha * q
            self.cg_r[i] -= alpha * self.cg_Ad[i]

        delta_old = delta_new
        delta_new = self.dot(self.cg_r, self.cg_r)
        beta = delta_new / delta_old

        for i in range(self.body.num_nodes):
            # d = r + beta * d
            self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]

        return delta_new

    @ti.kernel
    def cgUpdateVm(self):
        """依据共轭梯度法求解结果更新电压值

        :return:
        """
        for i in self.Vm:
            self.Vm[i] = self.cg_x[i]


@ti.data_oriented
class Electrophysiology_Aliec_Panfilov(Electrophysiology):
    """
    用于仿真心肌组织的电活动的类，其细胞级模型采用Aliec_Panfilov
    """

    def __init__(self, body: Body, sigma_f=1.1, sigma_s=1.0, sigma_n=1.0):
        """初始化Electrophysiology_Aliec_Panfilov类

        :param body: 物体的几何属性
        """

        # 父类Electrophysiology的构造方法
        super(Electrophysiology_Aliec_Panfilov, self).__init__(body, sigma_f, sigma_s, sigma_n)

        # Aliec_Panfilov 模型参数
        self.k = 8.0
        self.a = 0.01
        self.b = 0.15
        self.epsilon_0 = 0.04
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.C_m = 1.0

    @ti.func
    def calculate_Rv(self, i, dt):
        """
        dV_m/dt = kV_m/C_m (V_m + aV_m -V_m^2) + I_ext - (ka+w)/C_m * V_m
        y = V_m, q(y,t) = kV_m/C_m * (V_m + aV_m -V_m^2) + I_ext, p(y,t) = (ka+w)/C_m
        """
        self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)) + (
                self.k * self.Vm[i] / self.C_m * (self.Vm[i] * (1.0 + self.a - self.Vm[i])) + self.I_ext[i]) / (
                                 (self.k * self.a + self.w[i]) / self.C_m) * (
                                 1.0 - tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon(V_m, w) * k * V_m * (1 + b - V_m) - epsilon(V_m, w) * w
        epsilon(V_m, w) = epsilon_0 + mu_1 * w / (mu_2 + V_m)
        y = w, q(y,t) = epsilon(V_m, w) * k * V_m * (1 + b - V_m), p(y,t) = epsilon(V_m, w)
        """
        epsilon_Vm_w = self.epsilon_0 + self.mu_1 * self.w[i] / (self.mu_2 + self.Vm[i])
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * epsilon_Vm_w) + (
                self.k * self.Vm[i] * (1.0 + self.b - self.Vm[i])) * (
                               1.0 - tm.exp(-1.0 * dt * epsilon_Vm_w))

    @ti.func
    def calculate_Rz(self, i, dt):
        pass


@ti.data_oriented
class Electrophysiology_FitzHugh_Nagumo(Electrophysiology):
    """用于仿真心肌组织的电活动的类，其细胞级模型采用FitzHugh_Nagumo模型

    """

    def __init__(self, body: Body):
        """初始化 Electrophysiology_FitzHugh_Nagumo 类

        :param body: 物体的几何属性
        """

        # 父类Electrophysiology的构造方法
        super(Electrophysiology_FitzHugh_Nagumo, self).__init__(body)

        # FitzHugh_Nagumo 模型参数
        self.a = 0.1
        self.epsilon_0 = 0.01
        self.beta = 0.5
        self.gamma = 1.0
        self.sigma = 0.0
        self.C_m = 1.0

    @ti.func
    def calculate_Rv(self, i, dt):
        """
        dV_m/dt = 1/C_m * [V_m(V_m + aV_m - V_m^2) - w] - a/C_m * V_m
        y = V_m, q(y,t) = 1/C_m * [V_m(V_m + aV_m - V_m^2) - w], p(y,t) = a/C_m
        """
        self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * dt * (self.a / self.C_m)) + (
                1.0 / self.C_m * (self.Vm[i] * self.Vm[i] * (1.0 + self.a - self.Vm[i]) - self.w[i])) / (
                                 self.a / self.C_m) * (
                                 1.0 - tm.exp(-1.0 * dt * (self.a / self.C_m)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon_0 * (beta * V_m - sigma) - epsilon_0 * gamma * w
        y = w, q(y,t) = epsilon_0 * (beta * V_m - sigma), p(y,t) = epsilon_0 * gamma
        """
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * self.epsilon_0 * self.gamma) + (
                self.epsilon_0 * (self.beta * self.Vm[i] - self.sigma) / (self.epsilon_0 * self.gamma)) * (
                            1.0 - tm.exp(-1.0 * dt * (self.epsilon_0 * self.gamma)))

    @ti.func
    def calculate_Rz(self, i, dt):
        pass
