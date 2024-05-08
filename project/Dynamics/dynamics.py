import taichi as ti
import taichi.math as tm
import numpy as np
from project.Geometry import Body


@ti.data_oriented
class Dynamics_XPBD_SNH_Active:
    """动力学仿真类
        - 仿真方法: 基于位置的扩展动力学(XPBD)
        - 被动材料模型: Stable Neo-Hookean模型
        - 附带主动力
    属性:
        num_pts_np: 每个集合中四面体的个数
        dt: 时间步长
        numSubsteps: 子步数量
        numPosIters: 迭代次数
    """

    def __init__(self, body: Body, num_pts_np: np.ndarray,
                 dt=1. / 6. / 1.29, numSubsteps=1, numPosIters=1,
                 Youngs_modulus=17000.0, Poisson_ratio=0.45,
                 ):
        """Dynamics_XPBD_SNH_Active类初始化

        :param body: 物体的几何属性
        :param num_pts_np: 约束独立集合数量
        :param dt: 时间步长
        :param numSubsteps: 子步数量
        :param numPosIters: 迭代次数
        :param Youngs_modulus: 杨氏模量
        :param Poisson_ratio: 泊松比
        """

        # 几何属性
        self.body = body
        # 顶点数量
        self.nv = self.body.num_nodes
        self.nodes = self.body.nodes
        # 四面体数量
        self.ne = self.body.num_elements
        self.elements = self.body.elements

        # 时间步长
        self.dt = dt
        # 子步数量
        self.numSubsteps = numSubsteps
        # 子步时间步长
        self.h = self.dt / self.numSubsteps
        # 迭代次数
        self.numPosIters = numPosIters

        # Lame'参数: 材料参数
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        self.invLa = 1.0 / self.LameLa
        self.invMu = 1.0 / self.LameMu

        # --------------------------------------------XPBD辅助变量-------------------------------------------------------
        # 顶点质量
        self.mass = ti.field(float, shape=self.nv)
        self.invMass = ti.field(float, shape=self.nv)
        # 外力（施加在顶点上）
        self.f_ext = ti.Vector.field(3, float, shape=self.nv)
        # 重力
        self.gravity = tm.vec3(0.0, 0.0, 0.0)
        # 位置
        self.pos = self.body.nodes
        # 上一步位置
        self.prevPos = ti.Vector.field(3, float, shape=self.nv)
        # 速度
        self.vel = ti.Vector.field(3, float, shape=self.nv)
        # 位置增量
        self.dx = ti.Vector.field(3, float, shape=self.nv)
        # TODO: 四面体体积力学和电学应该是共用的，电学里面的形变梯度也应该可以共用
        # 四面体体积
        self.vol = self.body.volume
        self.invVol = ti.field(float, shape=self.ne)
        # XPBD辅助变量, 记录每步迭代中每个约束投影导致的每个四面体的四个顶点在空间中的投影量
        self.grads = ti.Vector.field(3, float, shape=(self.ne, 4))
        # TODO: 主动张力使用顶点值ver_Ta更新，加权得到网格值tet_Ta
        # 主动张力Ta(网格上)
        self.tet_Ta = body.tet_Ta
        # XPBD辅助变量: Lagrange乘子，每个约束一个
        self.Lagrange_multiplier = ti.field(float, shape=(self.ne, 4))
        self.init()

        # XPBD并行化所需的独立约束集合（预处理中完成）
        self.tol_tet_set = self.body.num_tet_set[None]
        self.num_pts = ti.field(int, shape=(self.tol_tet_set,))
        self.num_pts.from_numpy(num_pts_np)

    @ti.kernel
    def init(self):
        """
        动力学仿真类Dynamics_XPBD_SNH_Active的taichi field成员变量初始化
        """
        # 速度初始化为0
        for i in self.vel:
            self.vel[i] = tm.vec3([0., 0., 0.])

        for i in self.nodes:
            self.mass[i] = 0.0
            self.f_ext[i] = self.gravity

        for i in self.elements:
            self.invVol[i] = 1. / self.vol[i]
            pm = self.vol[i] / 4.0 * self.body.density
            vid = tm.ivec4([0, 0, 0, 0])
            for j in ti.static(range(4)):
                vid[j] = self.elements[i][j]
                self.mass[vid[j]] += pm

        for i in self.nodes:
            self.invMass[i] = 1.0 / self.mass[i]

        # TODO: 补充初始化

    def update(self):
        """
        XPBD迭代更新
        """
        # self.update_Ta(self.dt)
        for _ in range(self.numSubsteps):
            self.update_Ta(self.h)
            self.sub_step()

    @ti.kernel
    def update_Ta(self, dt: float):
        """
        更新主动张力Ta
        按照时间步长更新顶点主动张力Ta，网格主动张力tet_Ta由ver_Ta加权得到
        """
        epsilon_0 = 1
        # epsilon_0 = 10
        k_Ta = 47.9  # kPa
        for i in self.pos:
            V = self.body.Vm[i]
            epsilon = 10 * epsilon_0
            if V < 0.05:
                epsilon = epsilon_0
            Ta_old = self.body.ver_Ta[i]
            Ta_new = dt * epsilon * k_Ta * V + Ta_old
            Ta_new /= (1 + dt * epsilon)
            self.body.ver_Ta[i] = Ta_new

        for i in self.elements:
            vid = tm.ivec4(0, 0, 0, 0)
            ver_mass = tm.vec4(0, 0, 0, 0)
            sum_mass = 0.0
            for j in ti.static(range(4)):
                vid[j] = self.elements[i][j]
                ver_mass[j] = self.mass[vid[j]]
                sum_mass += ver_mass[j]
            self.tet_Ta[i] = 0.0
            for j in ti.static(range(4)):
                self.tet_Ta[i] += ver_mass[j] / sum_mass * self.body.ver_Ta[vid[j]]

    def sub_step(self):
        """
        子步过程
        """
        self.preSolve()
        self.solve_Gauss_Seidel_GPU()
        self.postSolve()

    @ti.kernel
    def preSolve(self):
        """
        XPBD迭代中每次约束更新的前处理
        """
        pos, vel = ti.static(self.pos, self.vel)
        # 外力
        for i in self.f_ext:
            self.f_ext[i] = self.gravity

        # TODO: Neumann边界条件（血液压力）
        # for i in self.bou_endo_lv_face:
        #     id0, id1, id2 = self.bou_endo_lv_face[i][0], self.bou_endo_lv_face[i][1], self.bou_endo_lv_face[i][2]
        #     vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
        #     p1 = vert1 - vert0
        #     p2 = vert2 - vert0
        #     n1 = tm.cross(p1, p2)
        #     self.normal_bou_endo_lv_face[i] = tm.normalize(n1)
        #     self.f_ext[id0] += 1.0 * self.p_endo_lv * self.normal_bou_endo_lv_face[i] / 3.0
        #     self.f_ext[id1] += 1.0 * self.p_endo_lv * self.normal_bou_endo_lv_face[i] / 3.0
        #     self.f_ext[id2] += 1.0 * self.p_endo_lv * self.normal_bou_endo_lv_face[i] / 3.0
        # for i in self.bou_endo_rv_face:
        #     id0, id1, id2 = self.bou_endo_rv_face[i][0], self.bou_endo_rv_face[i][1], self.bou_endo_rv_face[i][2]
        #     vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
        #     p1 = vert1 - vert0
        #     p2 = vert2 - vert0
        #     n1 = tm.cross(p1, p2)
        #     self.normal_bou_endo_rv_face[i] = tm.normalize(n1)
        #     self.f_ext[id0] += 1.0 * self.p_endo_rv * self.normal_bou_endo_rv_face[i] / 3.0
        #     self.f_ext[id1] += 1.0 * self.p_endo_rv * self.normal_bou_endo_rv_face[i] / 3.0
        #     self.f_ext[id2] += 1.0 * self.p_endo_rv * self.normal_bou_endo_rv_face[i] / 3.0

        # 仅考虑外力下的位置和速度更新
        for i in self.pos:
            self.prevPos[i] = pos[i]
            vel[i] += self.h * self.f_ext[i] * self.invMass[i]
            pos[i] += self.h * vel[i]

        # Lagrange乘子初始化
        for i in self.elements:
            for j in ti.static(range(4)):
                self.Lagrange_multiplier[i, j] = 0.0

    def solve_Gauss_Seidel_GPU(self):
        """
        并行化的Gauss_Seidel迭代
        """
        for _ in range(self.numPosIters):
            left, right = 0, 0
            for set_id in range(self.tol_tet_set):
                if set_id == 0:
                    left = 0
                    right = self.num_pts[0]
                else:
                    left += self.num_pts[set_id - 1]
                    right += self.num_pts[set_id]
                self.solve_elem_Gauss_Seidel_GPU(left, right)

        # TODO: Dirichlet边界条件处理
        # self.solve_dirichlet_boundary()

    @ti.kernel
    def solve_elem_Gauss_Seidel_GPU(self, left: int, right: int):
        pos, vel, tet, ir, g = ti.static(self.pos, self.vel, self.elements, self.body.DmInv, self.grads)
        for i in range(left, right):
            C = 0.0
            devCompliance = 1.0 * self.invMu
            volCompliance = 1.0 * self.invLa
            id = tm.ivec4(0, 0, 0, 0)
            for j in ti.static(range(4)):
                id[j] = tet[i][j]

            # 偏向能量：
            # Psi = mu / 2 * (tr(F^T @ F) - 3.0)
            # C = sqrt(tr(F^T @ F))
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            r_s = tm.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
                          + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
                          + v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])
            r_s_inv = 1.0 / r_s
            dCDdF = r_s_inv * F
            self.computedCdx(i, dCDdF)
            C = r_s
            self.applyToElem(i, C, devCompliance, 0)

            # 静水能量
            # Psi = lambda / 2 * (det(F) - 1.0 - mu / lambda)^2
            # C = det(F) - 1.0 - mu / lambda
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            dF0 = F_col1.cross(F_col2)
            dF1 = F_col2.cross(F_col0)
            dF2 = F_col0.cross(F_col1)
            dCHdF = tm.mat3(dF0, dF1, dF2)
            dCHdF = dCHdF.transpose()
            self.computedCdx(i, dCHdF)
            vol = F.determinant()
            C = vol - 1.0 - volCompliance / devCompliance
            self.applyToElem(i, C, volCompliance, 1)

            # 各向异性弹性应力：
            # Psi = mu / 2.0 * (f^T @ F^T @ F @ f - 1.0)^2
            # v1 = pos[id[1]] - pos[id[0]]
            # v2 = pos[id[2]] - pos[id[0]]
            # v3 = pos[id[3]] - pos[id[0]]
            # Ds = tm.mat3(v1, v2, v3)
            # Ds = Ds.transpose()
            # F = Ds @ ir[i]
            # f0 = self.body.tet_fiber[i]
            # f = F @ f0
            # C = f.dot(f) - 1
            # C_inv = 1.0 / C
            # dI5dF = 2.0 * F @ (f0.outer_product(f0))
            # self.computedCdx(i, dI5dF)
            # self.applyToElem(i, C, volCompliance * 10000, 2)
            # 调整各向异性约束大小
            # self.applyToElem(i, C, volCompliance * 10000, 2)

            # 主动应力：
            # Psi = Ta / 2 * (f^T @ F^T @ F @ f - 1.0)
            # C = sqrt(f^T @ F^T @ F @ f)
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            f0 = self.body.tet_fiber[i]
            f = F @ f0
            C = tm.sqrt(f.dot(f))
            C_inv = 1.0 / C
            dCadF = C_inv * F @ (f0.outer_product(f0))
            self.computedCdx(i, dCadF)

            if self.body.tet_Ta[i] > 0:
                self.applyToElem(i, C, 1.0 / self.body.tet_Ta[i], 3)

    @ti.func
    def computedCdx(self, elemNr, dCdF):
        """
        计算dC/dx导数，其是12*1的向量，即grads
        """
        g = ti.static(self.grads)
        # PFPx是一个三阶张量，展平为9 * 12的矩阵，通过列向量计算dIdx
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        # 向量化dIdF
        vec_dIdF = ti.Vector([dCdF[0, 0], dCdF[1, 0], dCdF[2, 0],
                              dCdF[0, 1], dCdF[1, 1], dCdF[2, 1],
                              dCdF[0, 2], dCdF[1, 2], dCdF[2, 2]], float)
        # 计算PFPx，参见章鱼书(course_dynamic_deformables)P180
        ir = ti.static(self.body.DmInv)
        m = ir[elemNr][0, 0]
        n = ir[elemNr][0, 1]
        o = ir[elemNr][0, 2]
        p = ir[elemNr][1, 0]
        q = ir[elemNr][1, 1]
        r = ir[elemNr][1, 2]
        s = ir[elemNr][2, 0]
        t = ir[elemNr][2, 1]
        u = ir[elemNr][2, 2]

        t1 = - m - p - s
        t2 = - n - q - t
        t3 = - o - r - u

        # vec0
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[0] = t1
        PFPx[3] = t2
        PFPx[6] = t3
        g[elemNr, 0][0] = PFPx.dot(vec_dIdF)

        # vec1
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[1] = t1
        PFPx[4] = t2
        PFPx[7] = t3
        g[elemNr, 0][1] = PFPx.dot(vec_dIdF)

        # vec2
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[2] = t1
        PFPx[5] = t2
        PFPx[8] = t3
        g[elemNr, 0][2] = PFPx.dot(vec_dIdF)

        # vec3
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        # if elemNr == 10:
        #     print("test1", PFPx)
        PFPx[0] = m
        PFPx[3] = n
        PFPx[6] = o
        g[elemNr, 1][0] = PFPx.dot(vec_dIdF)
        # if elemNr == 10:
        #     print("test2", vec_dIdF, m, n, o, g[elemNr, 1][0])
        #     print("test3", PFPx)

        # vec4
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[1] = m
        PFPx[4] = n
        PFPx[7] = o
        g[elemNr, 1][1] = PFPx.dot(vec_dIdF)

        # vec5
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[2] = m
        PFPx[5] = n
        PFPx[8] = o
        g[elemNr, 1][2] = PFPx.dot(vec_dIdF)

        # vec6
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[0] = p
        PFPx[3] = q
        PFPx[6] = r
        g[elemNr, 2][0] = PFPx.dot(vec_dIdF)

        # vec7
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[1] = p
        PFPx[4] = q
        PFPx[7] = r
        g[elemNr, 2][1] = PFPx.dot(vec_dIdF)

        # vec8
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[2] = p
        PFPx[5] = q
        PFPx[8] = r
        g[elemNr, 2][2] = PFPx.dot(vec_dIdF)

        # vec9
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[0] = s
        PFPx[3] = t
        PFPx[6] = u
        g[elemNr, 3][0] = PFPx.dot(vec_dIdF)

        # vec10
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[1] = s
        PFPx[4] = t
        PFPx[7] = u
        g[elemNr, 3][1] = PFPx.dot(vec_dIdF)

        # vec11
        PFPx = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        PFPx[2] = s
        PFPx[5] = t
        PFPx[8] = u
        g[elemNr, 3][2] = PFPx.dot(vec_dIdF)

    @ti.func
    def applyToElem(self, elemNr, C, compliance, cid):
        """
        单个约束更新
        """
        g, pos, elem, h, invVol, invMass = ti.static(self.grads, self.pos, self.elements, self.h, self.invVol,
                                                     self.invMass)
        w = 0.0
        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            w += (g[elemNr, i][0] * g[elemNr, i][0] + g[elemNr, i][1] * g[elemNr, i][1] + g[elemNr, i][2] *
                  g[elemNr, i][2]) * invMass[eid]

        dlambda = 0.0
        if w != 0.0:
            alpha = compliance / h / h * invVol[elemNr]
            dlambda = (0.0 - C - alpha * self.Lagrange_multiplier[elemNr, cid]) / (w + alpha)

        self.Lagrange_multiplier[elemNr, cid] += dlambda

        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            pos[eid] += g[elemNr, i] * (dlambda * invMass[eid])

    # @ti.kernel
    # def solve_dirichlet_boundary(self):
    #     """
    #     Dirichlet边界条件处理
    #     """
    #     for i in self.pos:
    #         if self.bou_dirichlet[i] == 1:
    #             self.pos[i][1] = self.prevPos[i][1]
    #             # self.pos[i] = self.prevPos[i]

    @ti.kernel
    def postSolve(self):
        """
        XPBD迭代中每次约束更新的后处理
        """
        pos, vel = ti.static(self.pos, self.vel)
        for i in pos:
            # 更新速度
            vel[i] = (pos[i] - self.prevPos[i]) / self.h
