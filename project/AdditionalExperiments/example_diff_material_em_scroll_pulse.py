import taichi as ti
import taichi.math as tm
import numpy as np
import project.Geometry.geometrytool as geo
from project.data.biventricular import meshData
import matplotlib.pyplot as plt


@ti.data_oriented
class Body:
    def __init__(self, vert_np: np.ndarray, tet_np: np.ndarray,
                 tet_fiber_np: np.ndarray, tet_sheet_np: np.ndarray, tet_normal_np: np.ndarray,
                 num_tet_set_np, tet_set_np: np.ndarray,
                 bou_base_np: np.ndarray, bou_epi_np: np.ndarray,
                 bou_endo_lv_np: np.ndarray, bou_endo_rv_np: np.ndarray
                 ) -> None:
        self.density = 1000.0
        self.num_vertex = len(vert_np)
        self.vertex = ti.Vector.field(3, dtype=float, shape=(self.num_vertex, ))
        self.vertex.from_numpy(vert_np)
        self.np_vertex = vert_np
        self.num_tet = len(tet_np)
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=(self.num_tet, ))
        self.elements.from_numpy(tet_np)
        self.np_elements = tet_np
        # tet_fiber
        self.tet_fiber = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_fiber.from_numpy(tet_fiber_np)
        # tet_sheet
        self.tet_sheet = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_sheet.from_numpy(tet_sheet_np)
        # tet_normal
        self.tet_normal = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_normal.from_numpy(tet_normal_np)
        # num_tet_set
        self.num_tet_set = ti.field(int, ())
        self.num_tet_set[None] = num_tet_set_np
        # tet_set
        self.tet_set = ti.field(int, shape=(self.num_tet,))
        self.tet_set.from_numpy(tet_set_np)

        # bou
        self.num_bou_base_face = len(bou_base_np)
        self.bou_base = ti.Vector.field(3, int, shape=(self.num_bou_base_face,))
        self.bou_base.from_numpy(bou_base_np)

        self.num_bou_epi_face = len(bou_epi_np)
        self.bou_epi = ti.Vector.field(3, int, shape=(self.num_bou_epi_face,))
        self.bou_epi.from_numpy(bou_epi_np)

        self.num_bou_endo_lv_face = len(bou_endo_lv_np)
        self.bou_endo_lv = ti.Vector.field(3, int, shape=(self.num_bou_endo_lv_face,))
        self.bou_endo_lv.from_numpy(bou_endo_lv_np)

        self.num_bou_endo_rv_face = len(bou_endo_rv_np)
        self.bou_endo_rv = ti.Vector.field(3, int, shape=(self.num_bou_endo_rv_face,))
        self.bou_endo_rv.from_numpy(bou_endo_rv_np)

        # variables for visualization
        surfaces = geo.get_surface_from_tet(nodes=vert_np, elements=tet_np)
        self.surfaces = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1]))
        self.surfaces.from_numpy(surfaces.reshape(-1))

        self.Dm = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.DmInvT = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.init_DmInv()

        self.vel = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.init_vel()
        self.Vm = ti.field(float, shape=(self.num_vertex,))

        # volume
        self.volume = ti.field(float, self.num_tet)
        self.init_volume()
        # # 顶点fiber方向
        # self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # self.vert_fiber.from_numpy(vert_fiber)

        self.tet_Ta = ti.field(float, shape=(self.num_tet,))
        self.ver_Ta = ti.field(float, shape=(self.num_vertex,))
        self.init_electrophysiology()

        self.vert_von_Mises = ti.field(float, shape=(self.num_vertex,))
        self.max_von_Mises = 50.0

        self.vert_color = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.init_vert_color()

        # 内外分层的标记
        # 1: endocardium (心内膜)
        # 2: mid-myocardium (中心肌)
        # 3: epicardium/septum (心外膜或隔膜)
        self.tag_layers_elements = ti.field(int, shape=(self.num_tet,))
        self.tag_layers_points = ti.field(int, shape=(self.num_vertex,))

    def get_near_vertex_index(self, x, y, z):
        min_dis = 100000.0
        res = -1
        for i in range(self.num_vertex):
            dis = (self.vertex[i][0] - x) * (self.vertex[i][0] - x) + \
                  (self.vertex[i][1] - y) * (self.vertex[i][1] - y) + \
                  (self.vertex[i][2] - z) * (self.vertex[i][2] - z)
            if dis < min_dis:
                min_dis = dis
                res = i
        print(res)

    @ti.kernel
    def update_color_Vm(self):
        for i in self.vert_color:
            self.vert_color[i] = tm.vec3([self.Vm[i], 0.0, 1 - self.Vm[i]])

    @ti.kernel
    def update_color_von_Mises(self):
        for i in self.vert_color:
            self.vert_color[i] = tm.vec3([self.vert_von_Mises[i] / self.max_von_Mises, 0.3, 0.5])

    @ti.kernel
    def init_vert_color(self):
        for i in self.vert_color:
            self.vert_color[i] = tm.vec3(250/255, 128/255, 114/255)

    @ti.kernel
    def init_electrophysiology(self):
        for i in self.elements:
            self.tet_Ta[i] = 60.0

        for i in self.vertex:
            self.ver_Ta[i] = 60.0 * 100  # 60.0 * 30

    @ti.kernel
    def set_Ta(self, value: float):
        for i in self.elements:
            self.tet_Ta[i] = value

        for i in self.vertex:
            self.ver_Ta[i] = value

    def get_min_y(self):
        min_y = 10000000.0
        # print("nv:", self.num_vertex)
        for i in range(self.num_vertex):
            # print("i:", i)
            if min_y > self.vertex[i][1]:
                min_y = self.vertex[i][1]
        print(min_y)
        return min_y

    @ti.kernel
    def scale_vertex(self, scale: float):
        for i in range(self.num_vertex):
            self.vertex[i] *= scale

    @ti.kernel
    def init_DmInv(self):
        Dm, vertex, tet = ti.static(self.Dm, self.vertex, self.elements)
        # 0412 -
        for i in range(self.num_tet):
            Dm[i][0, 0] = vertex[tet[i][1]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 0] = vertex[tet[i][1]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 0] = vertex[tet[i][1]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 1] = vertex[tet[i][2]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 1] = vertex[tet[i][2]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 1] = vertex[tet[i][2]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 2] = vertex[tet[i][3]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 2] = vertex[tet[i][3]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 2] = vertex[tet[i][3]][2] - vertex[tet[i][0]][2]

        for i in range(self.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()
            self.DmInvT[i] = self.DmInv[i].transpose()

    @ti.kernel
    def init_vel(self):
        for i in self.vel:
            self.vel[i] = tm.vec3([0., 0., 0.])

    @ti.kernel
    def init_volume(self):
        for i in self.volume:
            self.volume[i] = ti.abs(self.Dm[i].determinant()) / 6.0

    @ti.kernel
    def sample_tet_fiber(self):
        for i in range(self.num_tet):
            self.tet_fiber[i] = self.vert_fiber[self.elements[i][0]] + self.vert_fiber[self.elements[i][1]] + \
                                self.vert_fiber[self.elements[i][2]] + self.vert_fiber[self.elements[i][3]]
            self.tet_fiber[i] /= 4.0
            self.tet_fiber[i] /= tm.length(self.tet_fiber[i])

    @ti.kernel
    def translation(self, x: float, y: float, z: float):
        for i in self.vertex:
            self.vertex[i][0] += x
            self.vertex[i][1] += y
            self.vertex[i][2] += z

    @ti.kernel
    def color_vertex_with_tag_layers(self):
        for i in self.tag_layers_points:
            if self.tag_layers_points[i] == 1:
                self.vert_color[i] = tm.vec3(1, 0, 0)
            if self.tag_layers_points[i] == 2:
                self.vert_color[i] = tm.vec3(0, 1, 0)
            if self.tag_layers_points[i] == 3:
                self.vert_color[i] = tm.vec3(0, 0, 1)


@ti.data_oriented
class diffusion_reaction_multi_layer:
    """
    use 2nd-order Strang splitting: An intergrative smoothed particle hydrodynamics method for modeling cardiac function
    use Aliec-Panfilov model (single cell)
    """
    def __init__(self, body: Body):
        self.body = body
        # self.Vm = ti.field(float, shape=(body.num_vertex,))
        self.Vm = self.body.Vm
        self.w = ti.field(float, shape=(body.num_vertex,))
        self.I_ext = ti.field(float, shape=(body.num_vertex,))
        self.init_Vm_w_and_I()

        # parameter of Aliec-Panfilov model
        # 修改为不同材料，不同的参数
        self.k = ti.field(float, shape=(body.num_vertex,))
        self.a = ti.field(float, shape=(body.num_vertex,))
        self.b = ti.field(float, shape=(body.num_vertex,))
        self.epsilon_0 = ti.field(float, shape=(body.num_vertex,))
        self.mu_1 = ti.field(float, shape=(body.num_vertex,))
        self.mu_2 = ti.field(float, shape=(body.num_vertex,))
        self.init_elec_reaction_parameter()
        self.C_m = 1.0

        # parameter of diffusion model
        self.sigma_f = 1.1
        self.sigma_s = 1.0
        self.sigma_n = 1.0
        self.Dm = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Ds = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.F = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Be = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.init_Ds_F_Be()
        self.fiber = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.sheet = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.normal = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.init_fiber()
        self.DM = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Me = ti.Matrix.field(4, 4, float, shape=(body.num_tet,))
        self.Ke = ti.Matrix.field(4, 4, float, shape=(body.num_tet,))
        self.D = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))

        # For conjugate gradient method
        self.cg_x = ti.field(float, self.body.num_vertex)
        self.cg_Ax = ti.field(float, self.body.num_vertex)
        self.cg_b = ti.field(float, self.body.num_vertex)
        self.cg_r = ti.field(float, self.body.num_vertex)
        self.cg_d = ti.field(float, self.body.num_vertex)
        self.cg_Ad = ti.field(float, self.body.num_vertex)
        self.pcg_M = ti.field(float, self.body.num_vertex)
        self.pcg_s = ti.field(float, self.body.num_vertex)
        self.cg_epsilon = 1.0e-3
        # debug
        self.cg_A = ti.field(float, shape=(self.body.num_vertex, self.body.num_vertex))

        self.tag_s1 = ti.field(int, shape=(self.body.num_vertex,))
        self.tag_s2_1 = ti.field(int, shape=(self.body.num_vertex,))
        self.tag_s2_2 = ti.field(int, shape=(self.body.num_vertex,))
        self.init_sim_tag()

    @ti.kernel
    def init_elec_reaction_parameter(self):
        for i in self.body.tag_layers_points:
            if self.body.tag_layers_points[i] == 1:
                self.k[i] = 8.0
                self.a[i] = 0.01
                self.b[i] = 0.15
                self.epsilon_0[i] = 0.04
                self.mu_1[i] = 0.2
                self.mu_2[i] = 0.3
            elif self.body.tag_layers_points[i] == 2:
                self.k[i] = 16.0
                self.a[i] = 0.02
                self.b[i] = 0.30
                self.epsilon_0[i] = 0.08
                self.mu_1[i] = 0.4
                self.mu_2[i] = 0.6
            elif self.body.tag_layers_points[i] == 3:
                self.k[i] = 10.0
                self.a[i] = 0.005
                self.b[i] = 0.075
                self.epsilon_0[i] = 0.02
                self.mu_1[i] = 0.1
                self.mu_2[i] = 0.15

    @ti.kernel
    def init_sim_tag(self):
        pos = ti.static(self.body.vertex)
        for i in pos:
            if -24 <= pos[i].x <= -18.0 and -6.0 <= pos[i].y <= 0.0 and -3.0 <= pos[i].z <= 3.0:
                self.tag_s1[i] = 1
            else:
                self.tag_s1[i] = 0

            if 6.0 >= pos[i].x >= 0.0 >= pos[i].y >= -6.0 and 0.0 <= pos[i].z:
                self.tag_s2_1[i] = 1
            else:
                self.tag_s2_1[i] = 0

            if 12.0 >= pos[i].x >= 0.0 >= pos[i].y >= -24.0 and 0.0 <= pos[i].z:
                self.tag_s2_2[i] = 1
            else:
                self.tag_s2_2[i] = 0

    @ti.kernel
    def init_Vm_w_and_I(self):
        for i in self.Vm:
            self.Vm[i] = 0.0

        for i in self.w:
            self.w[i] = 0.0

        for i in self.I_ext:
            self.I_ext[i] = 0.0

    @ti.kernel
    def init_Ds_F_Be(self):
        for i in range(self.body.num_tet):
            self.Dm[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 0] = self.body.vertex[self.body.elements[i][0]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Dm[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 1] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Dm[i][0, 2] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 2] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 2] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]

        for i in range(self.body.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()

        for i in range(self.body.num_tet):
            self.Ds[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 0] = self.body.vertex[self.body.elements[i][0]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Ds[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 1] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Ds[i][0, 2] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 2] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 2] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]

        for i in range(self.body.num_tet):
            self.Be[i][0, 0] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 0] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 0] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]
            self.Be[i][0, 1] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 1] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 1] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]
            self.Be[i][0, 2] = self.body.vertex[self.body.elements[i][3]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 2] = self.body.vertex[self.body.elements[i][3]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 2] = self.body.vertex[self.body.elements[i][3]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]

        for i in range(self.body.num_tet):
            self.F[i] = self.Ds[i] @ self.DmInv[i]

    @ti.kernel
    def init_fiber(self):
        for i in range(self.body.num_tet):
            self.fiber[i] = self.F[i] @ self.body.tet_fiber[i]
            self.sheet[i] = self.F[i] @ self.body.tet_sheet[i]
            self.normal[i] = self.F[i] @ tm.cross(self.body.tet_fiber[i], self.body.tet_sheet[i])

    @ti.kernel
    def calculate_reaction(self, dt: float):
        """
        2 ODEs:
        Rv: C_m dV_m/dt = I_ion(V_m, w)
        Rw: dw/dt = g(V_m, w)
        use Reaction-by-Reaction splitting method to decouple ODEs,
        update: R(t+dt)=Rv(dt/2){Rw(dt/2){Rw(dt/2){Rv(dt/2){R(t)}}}}
        rewrite a single reaction equation in a general form as
        dy/dt = q(y,t) - p(y,t)y
        use QSS method, a linearly approximated exact solution, which is unconditionally stable:
        y(t+dt) = y(t)exp(-p(y(t),t)dt) + q(y(t),t)/p(y(t),t) * (1-exp(-p(y(t),t)dt))
        """
        for i in self.Vm:
            self.calculate_Rv(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rv(i, dt * 0.5)

    @ti.func
    def calculate_Rv(self, i, dt):
        """
        dV_m/dt = kV_m/C_m (V_m + aV_m -V_m^2) + I_ext - (ka+w)/C_m * V_m
        y = V_m, q(y,t) = kV_m/C_m * (V_m + aV_m -V_m^2) + I_ext, p(y,t) = (ka+w)/C_m
        """
        self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * dt * ((self.k[i] * self.a[i] + self.w[i]) / self.C_m)) + (
                self.k[i] * self.Vm[i] / self.C_m * (self.Vm[i] * (1.0 + self.a[i] - self.Vm[i])) + self.I_ext[i]) / (
                                 (self.k[i] * self.a[i] + self.w[i]) / self.C_m) * (
                                 1.0 - tm.exp(-1.0 * dt * ((self.k[i] * self.a[i] + self.w[i]) / self.C_m)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon(V_m, w) * k * V_m * (1 + b - V_m) - epsilon(V_m, w) * w
        epsilon(V_m, w) = epsilon_0 + mu_1 * w / (mu_2 + V_m)
        y = w, q(y,t) = epsilon(V_m, w) * k * V_m * (1 + b - V_m), p(y,t) = epsilon(V_m, w)
        """
        epsilon_Vm_w = self.epsilon_0[i] + self.mu_1[i] * self.w[i] / (self.mu_2[i] + self.Vm[i])
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * epsilon_Vm_w) + (
                self.k[i] * self.Vm[i] * (1.0 + self.b[i] - self.Vm[i])) * (
                               1.0 - tm.exp(-1.0 * dt * epsilon_Vm_w))

    @ti.kernel
    def calculate_M_and_K(self):
        fiber, sheet, normal = ti.static(self.fiber, self.sheet, self.normal)
        for i in range(self.body.num_tet):
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
            J = self.F[i].determinant()
            A = J_phi @ self.Be[i].inverse() @ self.F[i].inverse()
            self.Ke[i] = 1.0 / 6.0 * J * ti.abs(self.Be[i].determinant()) * A @ self.DM[i] @ A.transpose() * 6.0
            self.D[i] = J * self.F[i].inverse() @ self.DM[i] @ self.F[i].inverse().transpose()

    @ti.kernel
    def compute_RHS(self):
        for i in self.cg_b:
            self.cg_b[i] = 0.0
        # rhs = b = f * dt + M * u(t), here, f = 0
        for i in range(self.body.num_tet):
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

    @ti.func
    def A_mult_x(self, dt, dst, src):
        # lhs = Ax = (M + K * dt) * u(t+1)
        for i in range(self.body.num_vertex):
            dst[i] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
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
        result = 0.0
        for i in range(self.body.num_vertex):
            result += v1[i] * v2[i]
        return result

    # cg #
    # --------------------------------------------------------------------------------------------------- #
    @ti.kernel
    def cg_before_ite(self, dt: float) -> float:
        for i in range(self.body.num_vertex):
            self.cg_x[i] = self.Vm[i]
        self.A_mult_x(dt, self.cg_Ax, self.cg_x)

        for i in range(self.body.num_vertex):
            # r = b - A @ x
            self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]
            # d = r
            self.cg_d[i] = self.cg_r[i]

        delta_new = self.dot(self.cg_r, self.cg_r)
        return delta_new

    @ti.kernel
    def cg_run_iteration(self, dt: float, delta: float) -> float:
        delta_new = delta
        # q = A @ d
        self.A_mult_x(dt, self.cg_Ad, self.cg_d)
        # alpha = delta_new / d.dot(q)
        alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)

        for i in range(self.body.num_vertex):
            # x = x + alpha * d
            self.cg_x[i] += alpha * self.cg_d[i]
            # r = b - A @ x || r = r - alpha * q
            self.cg_r[i] -= alpha * self.cg_Ad[i]
        delta_old = delta_new
        delta_new = self.dot(self.cg_r, self.cg_r)
        beta = delta_new / delta_old
        for i in range(self.body.num_vertex):
            # d = r + beta * d
            self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
        return delta_new

    def cg(self, dt: float):
        delta_new = self.cg_before_ite(dt)
        delta_0 = delta_new
        ite, iteMax = 0, 100
        while ite < iteMax and delta_new > (self.cg_epsilon**2) * delta_0:
            delta_new = self.cg_run_iteration(dt, delta_new)
            ite += 1

    @ti.kernel
    def cgUpdateVm(self):
        for i in self.Vm:
            self.Vm[i] = self.cg_x[i]

    def calculate_diffusion(self, dt):
        self.calculate_M_and_K()
        self.compute_RHS()
        self.cg(dt)
        self.cgUpdateVm()

    def update_Vm(self, dt):
        self.calculate_reaction(dt * 0.5)
        self.calculate_diffusion(dt)
        self.calculate_reaction(dt * 0.5)

    @ti.kernel
    def apply_stimulation_s1(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if self.tag_s1[i] == 1:
                self.Vm[i] = 1.52

    @ti.kernel
    def apply_stimulation_s2_1(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if self.tag_s2_1[i] == 1:
                self.Vm[i] = 1.55

    @ti.kernel
    def apply_stimulation_s2_2(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if self.tag_s2_2[i] == 1:
                self.Vm[i] = 1.55

    @ti.kernel
    def get_near_vertex_index(self, x: float, y: float, z: float) -> int:
        vert = ti.static(self.body.vertex)
        res = 0
        for i in vert:
            if (vert[i][0] - x)**2 + (vert[i][1] - y)**2 + (vert[i][2] - z)**2 < 1e-2:
                res = i
                print(i)
        return res

    def update(self, sub_steps):
        dt = 1. / 1.29 / 6. / sub_steps
        # dt = 1. / 60. / sub_steps
        # dt = 100.0 / 12.9 / 6.0 / sub_steps
        for _ in range(sub_steps):
            self.update_Vm(dt)

        # self.update_color()

    @ti.kernel
    def update_color(self):
        color = ti.static(self.body.vert_color)
        for i in color:
            color[i] = tm.vec3([self.Vm[i], self.Vm[i], 1])

    @ti.kernel
    def calculate_A(self, dt: float):
        for i, j in self.cg_A:
            self.cg_A[i, j] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.cg_A[id0, id0] += self.Me[i][0, 0] + self.Ke[i][0, 0] * dt
            self.cg_A[id0, id1] += self.Me[i][0, 1] + self.Ke[i][0, 1] * dt
            self.cg_A[id0, id2] += self.Me[i][0, 2] + self.Ke[i][0, 2] * dt
            self.cg_A[id0, id3] += self.Me[i][0, 3] + self.Ke[i][0, 3] * dt

            self.cg_A[id1, id0] += self.Me[i][1, 0] + self.Ke[i][1, 0] * dt
            self.cg_A[id1, id1] += self.Me[i][1, 1] + self.Ke[i][1, 1] * dt
            self.cg_A[id1, id2] += self.Me[i][1, 2] + self.Ke[i][1, 2] * dt
            self.cg_A[id1, id3] += self.Me[i][1, 3] + self.Ke[i][1, 3] * dt

            self.cg_A[id2, id0] += self.Me[i][2, 0] + self.Ke[i][2, 0] * dt
            self.cg_A[id2, id1] += self.Me[i][2, 1] + self.Ke[i][2, 1] * dt
            self.cg_A[id2, id2] += self.Me[i][2, 2] + self.Ke[i][2, 2] * dt
            self.cg_A[id2, id3] += self.Me[i][2, 3] + self.Ke[i][2, 3] * dt

            self.cg_A[id3, id0] += self.Me[i][3, 0] + self.Ke[i][3, 0] * dt
            self.cg_A[id3, id1] += self.Me[i][3, 1] + self.Ke[i][3, 1] * dt
            self.cg_A[id3, id2] += self.Me[i][3, 2] + self.Ke[i][3, 2] * dt
            self.cg_A[id3, id3] += self.Me[i][3, 3] + self.Ke[i][3, 3] * dt


@ti.data_oriented
class XPBD_SNH_with_active:
    def __init__(self, body: Body, num_pts_np: np.ndarray,
                 # Youngs_modulus=17000.0, Poisson_ratio=0.45,
                 Youngs_modulus=1000.0, Poisson_ratio=0.49,
                 # dt=1 / 50.0, numSubsteps=1, numPosIters=1):
                 dt = 1. / 6. / 1.29, numSubsteps = 1, numPosIters = 1):
        self.body = body
        self.num_vertex = self.body.num_vertex
        self.num_element = self.body.num_tet
        self.dt = dt
        self.numSubsteps = numSubsteps
        self.h = self.dt / self.numSubsteps
        self.numPosIters = numPosIters
        self.friction = 1000.0
        self.YoungsModulus = ti.field(float, shape=(body.num_tet,))
        self.PoissonRatio = ti.field(float, shape=(body.num_tet,))
        self.LameLa = ti.field(float, shape=(body.num_tet,))
        self.LameMu = ti.field(float, shape=(body.num_tet,))
        self.invLa = ti.field(float, shape=(body.num_tet,))
        self.invMu = ti.field(float, shape=(body.num_tet,))
        self.init_Lame_parameter()
        # self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        # self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        # self.invLa = 1.0 / self.LameLa
        # self.invMu = 1.0 / self.LameMu
        self.mass = ti.field(float, shape=(self.num_vertex,))
        self.f_ext = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.gravity = tm.vec3(0.0, 0.0, 0.0)
        self.pos = self.body.vertex
        self.prevPos = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.vel = self.body.vel
        self.dx = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.elements = self.body.elements
        self.invMass = ti.field(float, shape=(self.num_vertex,))
        self.vol = self.body.volume
        self.invVol = ti.field(float, shape=(self.num_element,))
        self.grads = ti.Vector.field(3, float, shape=(self.num_element, 4))
        self.tol_tet_set = self.body.num_tet_set[None]
        self.num_pts = ti.field(int, shape=(self.tol_tet_set,))
        self.num_pts.from_numpy(num_pts_np)
        self.tet_Ta = body.tet_Ta
        self.Lagrange_multiplier = ti.field(float, shape=(self.num_element, 4))
        self.init()

        # bou
        self.bou_dirichlet = ti.field(int, shape=(self.num_vertex, ))
        self.init_dirichlet_bou()

        self.is_cal_von_Mises = 0
        self.tet_von_Mises_stress = ti.field(float, shape=(self.num_element,))
        self.Cauchy_Stree = ti.Matrix.field(3, 3, float, shape=(self.num_element,))
        self.tol_vol_weight = ti.field(float, shape=(self.num_vertex, ))

        self.num_bou_endo_lv_face = self.body.num_bou_endo_lv_face
        self.bou_endo_lv_face = self.body.bou_endo_lv
        self.normal_bou_endo_lv_face = ti.Vector.field(3, float, shape=(self.num_bou_endo_lv_face,))
        self.num_bou_endo_rv_face = self.body.num_bou_endo_rv_face
        self.bou_endo_rv_face = self.body.bou_endo_rv
        self.normal_bou_endo_rv_face = ti.Vector.field(3, float, shape=(self.num_bou_endo_rv_face,))
        self.get_bou_face_normal()
        self.p_endo_lv = 0.0  # 15.0
        self.p_endo_rv = 0.0  # 15.0

        # self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # self.vert_fiber.from_numpy(vert_fiber_np)
        # self.F = ti.Matrix.field(3, 3, float, shape=(self.num_element,))

    def init_dirichlet_bou(self):
        for i in range(self.body.num_bou_base_face):
            vid0, vid1, vid2 = self.body.bou_base[i][0], self.body.bou_base[i][1], self.body.bou_base[i][2]
            self.bou_dirichlet[vid0] = 1
            self.bou_dirichlet[vid1] = 1
            self.bou_dirichlet[vid2] = 1

    @ti.kernel
    def init_Lame_parameter(self):
        for i in self.body.elements:
            if self.body.tag_layers_elements[i] == 1:
                self.YoungsModulus[i] = 17000.0
                self.PoissonRatio[i] = 0.45
            if self.body.tag_layers_elements[i] == 2:
                self.YoungsModulus[i] = 20000.0
                self.PoissonRatio[i] = 0.4
            if self.body.tag_layers_elements[i] == 3:
                self.YoungsModulus[i] = 30000.0
                self.PoissonRatio[i] = 0.49

            self.LameLa[i] = self.YoungsModulus[i] * self.PoissonRatio[i] / (
                        (1 + self.PoissonRatio[i]) * (1 - 2 * self.PoissonRatio[i]))
            self.LameMu[i] = self.YoungsModulus[i] / (2 * (1 + self.PoissonRatio[i]))
            self.invLa[i] = 1.0 / self.LameLa[i]
            self.invMu[i] = 1.0 / self.LameMu[i]

    @ti.kernel
    def get_bou_face_normal(self):
        for i in self.bou_endo_lv_face:
            id0, id1, id2 = self.bou_endo_lv_face[i][0], self.bou_endo_lv_face[i][1], self.bou_endo_lv_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_endo_lv_face[i] = tm.normalize(n1)

        for i in self.bou_endo_rv_face:
            id0, id1, id2 = self.bou_endo_rv_face[i][0], self.bou_endo_rv_face[i][1], self.bou_endo_rv_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_endo_rv_face[i] = tm.normalize(n1)

        # for i in self.bou_epi_face:
        #     id0, id1, id2 = self.bou_epi_face[i][0], self.bou_epi_face[i][1], self.bou_epi_face[i][2]
        #     vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
        #     p1 = vert1 - vert0
        #     p2 = vert2 - vert0
        #     n1 = tm.cross(p1, p2)
        #     self.normal_bou_epi_face[i] = tm.normalize(n1)

    @ti.kernel
    def init(self):
        for i in self.pos:
            self.mass[i] = 0.0
            self.f_ext[i] = self.gravity

        for i in self.elements:
            self.invVol[i] = 1. / self.vol[i]
            pm = self.vol[i] / 4.0 * self.body.density
            vid = tm.ivec4([0, 0, 0, 0])
            for j in ti.static(range(4)):
                vid[j] = self.elements[i][j]
                self.mass[vid[j]] += pm

        for i in self.pos:
            self.invMass[i] = 1.0 / self.mass[i]

    def update(self):
        self.update_Ta()
        for _ in range(self.numSubsteps):
            self.sub_step()

    @ti.kernel
    def update_Ta(self):
        epsilon_0 = 1
        k_Ta = 47.9 # kPa
        for i in self.pos:
            V = self.body.Vm[i]
            epsilon = 10 * epsilon_0
            if V < 0.05:
                epsilon = epsilon_0
            Ta_old = self.body.ver_Ta[i]
            Ta_new = self.dt * epsilon * k_Ta * V + Ta_old
            Ta_new /= (1 + self.dt * epsilon)
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
        self.preSolve()
        self.solve_Gauss_Seidel_GPU()
        self.postSolve()

    @ti.kernel
    def preSolve(self):
        pos, vel = ti.static(self.pos, self.vel)
        for i in self.f_ext:
            self.f_ext[i] = self.gravity

        for i in self.bou_endo_lv_face:
            id0, id1, id2 = self.bou_endo_lv_face[i][0], self.bou_endo_lv_face[i][1], self.bou_endo_lv_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_endo_lv_face[i] = tm.normalize(n1)
            self.f_ext[id0] += 1.0 * self.p_endo_lv * self.normal_bou_endo_lv_face[i] / 3.0
            self.f_ext[id1] += 1.0 * self.p_endo_lv * self.normal_bou_endo_lv_face[i] / 3.0
            self.f_ext[id2] += 1.0 * self.p_endo_lv * self.normal_bou_endo_lv_face[i] / 3.0

        for i in self.bou_endo_rv_face:
            id0, id1, id2 = self.bou_endo_rv_face[i][0], self.bou_endo_rv_face[i][1], self.bou_endo_rv_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_endo_rv_face[i] = tm.normalize(n1)
            self.f_ext[id0] += 1.0 * self.p_endo_rv * self.normal_bou_endo_rv_face[i] / 3.0
            self.f_ext[id1] += 1.0 * self.p_endo_rv * self.normal_bou_endo_rv_face[i] / 3.0
            self.f_ext[id2] += 1.0 * self.p_endo_rv * self.normal_bou_endo_rv_face[i] / 3.0

        for i in self.pos:
            self.prevPos[i] = pos[i]
            vel[i] += self.h * self.f_ext[i] * self.invMass[i]
            pos[i] += self.h * vel[i]

        for i in self.elements:
            for j in ti.static(range(4)):
                self.Lagrange_multiplier[i, j] = 0.0

    @ti.kernel
    def postSolve(self):
        pos, vel = ti.static(self.pos, self.vel)

        for i in pos:
            vel[i] = (pos[i] - self.prevPos[i]) / self.h

    def solve_Gauss_Seidel_GPU(self):
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

        self.solve_dirichlet_boundary()

    @ti.kernel
    def solve_dirichlet_boundary(self):
        for i in self.pos:
            if self.bou_dirichlet[i] == 1:
                self.pos[i][1] = self.prevPos[i][1]
                # self.pos[i] = self.prevPos[i]

    @ti.func
    def computedCdx(self, elemNr, dCdF):
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

    @ti.kernel
    def solve_elem_Gauss_Seidel_GPU(self, left: int, right: int):
        pos, vel, tet, ir, g = ti.static(self.pos, self.vel, self.elements, self.body.DmInv, self.grads)
        for i in range(left, right):
            C = 0.0
            devCompliance = 1.0 * self.invMu[i]
            volCompliance = 1.0 * self.invLa[i]
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
    def applyToElem(self, elemNr, C, compliance, cid):
        g, pos, elem, h, invVol, invMass = ti.static(self.grads, self.pos, self.elements, self.h, self.invVol,
                                                     self.invMass)
        w = 0.0
        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            w += (g[elemNr, i][0] * g[elemNr, i][0] + g[elemNr, i][1] * g[elemNr, i][1] + g[elemNr, i][2] * g[elemNr, i][2]) * invMass[eid]

        dlambda = 0.0
        if w != 0.0:
            alpha = compliance / h / h * invVol[elemNr]
            dlambda = (0.0 - C - alpha * self.Lagrange_multiplier[elemNr, cid]) / (w + alpha)

        self.Lagrange_multiplier[elemNr, cid] += dlambda

        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            pos[eid] += g[elemNr, i] * (dlambda * invMass[eid])

    @ti.kernel
    def cal_von_Mises(self):
        pos, tet, ir, cs = ti.static(self.pos, self.elements, self.body.DmInv, self.Cauchy_Stree)
        for i in self.body.vert_von_Mises:
            self.body.vert_von_Mises[i] = 0.0
        for i in self.elements:
            vid = tm.ivec4(0, 0, 0, 0)
            for j in ti.static(range(4)):
                vid[j] = tet[i][j]
            v1 = pos[vid[1]] - pos[vid[0]]
            v2 = pos[vid[2]] - pos[vid[0]]
            v3 = pos[vid[3]] - pos[vid[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]

            # P_p = lambda * (d I_3) / (d F) + mu / 2 * (d I_2) / (d F)
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            dI3_col0 = F_col1.cross(F_col2)
            dI3_col1 = F_col2.cross(F_col0)
            dI3_col2 = F_col0.cross(F_col1)
            dI3dF = tm.mat3((dI3_col0[0], dI3_col1[0], dI3_col2[0]),
                            (dI3_col0[1], dI3_col1[1], dI3_col2[1]),
                            (dI3_col0[2], dI3_col1[2], dI3_col2[2]))
            Pp = self.LameLa[i] * dI3dF + self.LameMu[i] * F
            # Pa = T_a * F * f0 * f0^T
            f0 = self.body.tet_fiber[i]
            Ff0 = F @ f0
            Pa = self.tet_Ta[i] * Ff0.outer_product(f0)
            P = Pp + Pa
            cs[i] = 1.0 / (F.determinant()) * P @ F.transpose()
            self.tet_von_Mises_stress[i] = tm.sqrt(0.5 * ((cs[i][0, 0] - cs[i][1, 1])**2 +
                                                          (cs[i][1, 1] - cs[i][2, 2])**2 +
                                                          (cs[i][2, 2] - cs[i][0, 0])**2 +
                                                          6.0 * (cs[i][0, 1] * cs[i][0, 1] +
                                                                 cs[i][1, 2] * cs[i][1, 2] +
                                                                 cs[i][2, 0] * cs[i][2, 0])))
            vol_elem = abs(Ds.determinant()) / 6.0
            for j in ti.static(range(4)):
                self.body.vert_von_Mises[vid[j]] += self.tet_von_Mises_stress[i] / 4.0
                # self.body.vert_von_Mises[vid[j]] += self.tet_von_Mises_stress[i] * vol_elem
                # self.tol_vol_weight[vid[j]] += vol_elem


def read_data():
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))
    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # tet_normal方向
    normal_tet_np = np.array(meshData['normalDirection'], dtype=float)
    normal_tet_np = normal_tet_np.reshape((-1, 3))
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # bou_tag
    bou_base_face_np = np.array(meshData['bou_base_face'], dtype=int)
    bou_base_face_np = bou_base_face_np.reshape((-1, 3))
    bou_endo_lv_face_np = np.array(meshData['bou_endo_lv_face'], dtype=int)
    bou_endo_lv_face_np = bou_endo_lv_face_np.reshape((-1, 3))
    bou_endo_rv_face_np = np.array(meshData['bou_endo_rv_face'], dtype=int)
    bou_endo_rv_face_np = bou_endo_rv_face_np.reshape((-1, 3))
    bou_epi_face_np = np.array(meshData['bou_epi_face'], dtype=int)
    bou_epi_face_np = bou_epi_face_np.reshape((-1, 3))

    Body_ = Body(vert_np=pos_np,
                 tet_np=tet_np,
                 tet_fiber_np=fiber_tet_np,
                 tet_sheet_np=sheet_tet_np,
                 tet_normal_np=normal_tet_np,
                 num_tet_set_np=num_tet_set_np,
                 tet_set_np=tet_set_np,
                 bou_base_np=bou_base_face_np,
                 bou_epi_np=bou_epi_face_np,
                 bou_endo_lv_np=bou_endo_lv_face_np,
                 bou_endo_rv_np=bou_endo_rv_face_np
                 )

    return Body_


def example_em_scroll_pulse_multi_layer():
    body = read_data()
    # 心肌分层标记
    mark_tag_layers(body)
    # 为不同标记的点着色
    body.color_vertex_with_tag_layers()

    ep_sys = diffusion_reaction_multi_layer(body=body)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_sys = XPBD_SNH_with_active(body=body, num_pts_np=num_per_tet_set_np)
    dynamics_sys.is_cal_von_Mises = 0

    open_gui = True
    # set parameter
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    if open_gui:
        # init the window, canvas, scene and camera
        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((1., 1., 1.))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # initial camera position
        camera.position(-10.50518621, 40.032794, 117.36565294)
        camera.lookat(-10.47637129, 39.56587387, 116.48182304)
        camera.up(0., 1., 0.)

        iter_time = 0
        while window.running:
            if iter_time == 0:
                ep_sys.apply_stimulation_s1()
                iter_time += 1
            elif iter_time == 290:
                ep_sys.apply_stimulation_s2_1()
                iter_time += 1
            else:
                iter_time += 1

            ep_sys.update(1)
            dynamics_sys.update()
            if dynamics_sys.is_cal_von_Mises == 1:
                dynamics_sys.cal_von_Mises()
                body.update_color_von_Mises()
            else:
                body.update_color_Vm()

            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, per_vertex_color=ep_sys.body.vert_color)
            # scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, color=(1.0, 0.5, 0.5))

            # show the frame
            canvas.scene(scene)
            window.show()
            # filename = f'frame_{i:05d}.png'  # create filename with suffix png
            # print(f'Frame {i} is recorded in {filename}')
            # gui.show(filename)  # export and show in GUI

    # 导出vtk
    # import tool.vtktool
    # for i in range(1000):
    #     if i == 0:
    #         ep_sys.apply_stimulation_s1()
    #     elif i == 290:
    #         ep_sys.apply_stimulation_s2_1()
    #     ep_sys.update(1)
    #     dynamics_sys.update()
    #     if i % 100 == 0:
    #         filename = "../data/res/example4/diff_material/%03d.vtk" % i
    #         np_nodes = body.vertex.to_numpy()
    #         np_elements = body.elements.to_numpy()
    #         np_Vm = body.Vm.to_numpy()
    #         for j in range(body.num_vertex):
    #             np_Vm[j] = 100.0 * np_Vm[j] - 80.0
    #         tool.vtktool.write_vtk(filename, np_nodes, np_elements, np_Vm)


@ti.kernel
def mark_tag_layers(body: ti.template()):
    # 初始化心内膜心外膜标记
    for i in body.tag_layers_points:
        body.tag_layers_points[i] = 0
    for i in body.tag_layers_elements:
        body.tag_layers_elements[i] = 0
    for i in body.bou_endo_lv:
        for j in ti.static(range(3)):
            body.tag_layers_points[body.bou_endo_lv[i][j]] = 1
    for i in body.bou_endo_rv:
        for j in ti.static(range(3)):
            body.tag_layers_points[body.bou_endo_rv[i][j]] = 1
    for i in body.bou_epi:
        for j in ti.static(range(3)):
            body.tag_layers_points[body.bou_epi[i][j]] = 3

    for i in body.elements:
        vid0, vid1, vid2, vid3 = body.elements[i][0], body.elements[i][1], body.elements[i][2], body.elements[i][3]
        if body.tag_layers_points[vid0] == 1 \
                or body.tag_layers_points[vid1] == 1 \
                or body.tag_layers_points[vid2] == 1 \
                or body.tag_layers_points[vid3] == 1:
            body.tag_layers_elements[i] = 1
        if body.tag_layers_points[vid0] == 3 \
                or body.tag_layers_points[vid1] == 3 \
                or body.tag_layers_points[vid2] == 3 \
                or body.tag_layers_points[vid3] == 3:
            body.tag_layers_elements[i] = 3

    for i in body.elements:
        if body.tag_layers_elements[i] == 1:
            for j in range(4):
                body.tag_layers_points[body.elements[i][j]] = 1
    for i in body.elements:
        if body.tag_layers_elements[i] == 3:
            for j in range(4):
                body.tag_layers_points[body.elements[i][j]] = 3

    # 标记中心肌
    for i in body.elements:
        if body.tag_layers_elements[i] == 0:
            body.tag_layers_elements[i] = 2
    for i in body.vertex:
        if body.tag_layers_points[i] == 0:
            body.tag_layers_points[i] = 2


def multi_layer_myocardium():
    body = read_data()
    # 心肌分层标记
    mark_tag_layers(body)
    # 为不同标记的点着色
    body.color_vertex_with_tag_layers()

    # 导出vtk
    import tool.vtktool
    filename = "../data/res/example4/diff_material/origin.vtk"
    np_nodes = body.vertex.to_numpy()
    np_elements = body.elements.to_numpy()
    np_Vm = body.Vm.to_numpy()
    for j in range(body.num_vertex):
        np_Vm[j] = body.tag_layers_points[j]
    tool.vtktool.write_vtk(filename, np_nodes, np_elements, np_Vm)

    # 可视化调试
    open_gui = False
    if open_gui:
        # 设置窗口参数
        windowLength = 1024
        lengthScale = min(windowLength, 512)
        light_distance = lengthScale / 25.
        # 初始化 window, canvas, scene and camera
        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((1., 1., 1.))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # initial camera position
        camera.position(-10.50518621, 40.032794, 117.36565294)
        camera.lookat(-10.47637129, 39.56587387, 116.48182304)
        camera.up(0., 1., 0.)

        while window.running:
            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, per_vertex_color=body.vert_color)
            # scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, color=(1.0, 0.5, 0.5))

            # show the frame
            canvas.scene(scene)
            window.show()


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    # ti.init(arch=ti.cpu, default_fp=ti.f64)

    # 3D em couple
    # example_em_scroll_pulse()
    # get_gif()

    # 生成多层心肌
    # multi_layer_myocardium()
    example_em_scroll_pulse_multi_layer()

