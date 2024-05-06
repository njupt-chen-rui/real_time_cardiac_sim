import taichi as ti
import taichi.math as tm
import numpy as np
import project.Geometry.geometrytool as geo_tool
import project.tool.colormap as colormap


@ti.data_oriented
class Body:
    """物体基本属性类

    属性:
        nodes: 物体的三维顶点坐标
        elements: 物体的四面体顶点索引
        density: 物体密度，单位: kg/m^3
        tet_fiber: 网格上的fiber方向
        tet_sheet: 网格上的sheet方向
        tet_normal: 网格上的normal方向
        surfaces: 网格表面（用于可视化）
        nodes_color: 顶点颜色，用于表示属性
        Vm: 顶点电压
    """

    def __init__(self, nodes_np: np.ndarray, elements_np: np.ndarray,
                 tet_fiber_np: np.ndarray, tet_sheet_np: np.ndarray, tet_normal_np: np.ndarray,
                 num_tet_set_np, tet_set_np: np.ndarray,
                 density=1120.0) -> None:
        """使用外界读入的numpy数组初始化Body类

        Args:
            nodes_np: 物体的三维顶点坐标
            elements_np: 物体的四面体网格
            tet_fiber_np: 网格fiber方向
            tet_sheet_np: 网格sheet方向
            tet_normal_np: 网格normal方向
            num_tet_set_np: 约束独立集合的个数
            tet_set_np: 约束独立集
            density(float): 密度
        """

        # 读入顶点数据
        self.num_nodes = len(nodes_np)
        """顶点数"""
        self.nodes = ti.Vector.field(3, float, shape=self.num_nodes)
        self.nodes.from_numpy(nodes_np)

        # 读入有限单元数据
        self.num_elements = len(elements_np)
        """单元数"""
        self.elements = ti.Vector.field(4, int, shape=self.num_elements)
        self.elements.from_numpy(elements_np)

        # 密度：kg/m^3
        self.density = density

        # 纤维方向(网格上的)
        self.tet_fiber = ti.Vector.field(3, dtype=float, shape=self.num_elements)
        self.tet_fiber.from_numpy(tet_fiber_np)

        self.tet_sheet = ti.Vector.field(3, dtype=float, shape=self.num_elements)
        self.tet_sheet.from_numpy(tet_sheet_np)

        self.tet_normal = ti.Vector.field(3, dtype=float, shape=self.num_elements)
        self.tet_normal.from_numpy(tet_normal_np)

        # TODO: 四面体形变，力学和电学应该可以统一使用
        self.Dm = ti.Matrix.field(3, 3, float, shape=self.num_elements)
        self.DmInv = ti.Matrix.field(3, 3, float, shape=self.num_elements)
        self.DmInvT = ti.Matrix.field(3, 3, float, shape=self.num_elements)
        self.init_DmInv()

        # 四面体体积(原始体积)
        self.volume = ti.field(float, self.num_elements)
        self.init_volume()

        # 顶点电压
        self.Vm = ti.field(float, shape=self.num_nodes)

        # 主动应力
        self.tet_Ta = ti.field(float, shape=self.num_elements)
        self.ver_Ta = ti.field(float, shape=self.num_nodes)
        self.init_Ta()

        # TODO: 将约束集合划分转移到XPBD模型中，而不是放在几何模型中
        # num_tet_set: 网格独立集合总数
        # TODO: 数字应该可以转变为一个数据，无需taichi.field
        self.num_tet_set = ti.field(int, ())
        self.num_tet_set[None] = num_tet_set_np
        # tet_set: 网格所属独立集合索引
        self.tet_set = ti.field(int, shape=self.num_elements)
        self.tet_set.from_numpy(tet_set_np)

        # TODO:边界

        # 计算网格表面, 用于辅助可视化
        surfaces = geo_tool.get_surface_from_tet(nodes=nodes_np, elements=elements_np)
        self.surfaces = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1]))
        self.surfaces.from_numpy(surfaces.reshape(-1))

        # 每个顶点的颜色
        self.nodes_color = ti.Vector.field(3, float, shape=self.num_nodes)
        self.init_nodes_color()

    @ti.kernel
    def init_DmInv(self):
        """求Dm和其逆矩阵，用于计算形变梯度

        |Dm[0, 0], Dm[0, 1], Dm[0, 2]|   |(v1.x - v0.x), (v2.x - v0.x), (v3.x - v0.x)|
        |Dm[1, 0], Dm[1, 1], Dm[1, 2]| = |(v1.y - v0.y), (v2.y - v0.y), (v3.y - v0.y)|
        |Dm[2, 0], Dm[2, 1], Dm[2, 2]|   |(v1.z - v0.z), (v2.z - v0.z), (v3.z - v0.z)|

        :return:
        """

        Dm, vertex, tet = ti.static(self.Dm, self.nodes, self.elements)
        for i in range(self.num_elements):
            Dm[i][0, 0] = vertex[tet[i][1]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 0] = vertex[tet[i][1]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 0] = vertex[tet[i][1]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 1] = vertex[tet[i][2]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 1] = vertex[tet[i][2]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 1] = vertex[tet[i][2]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 2] = vertex[tet[i][3]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 2] = vertex[tet[i][3]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 2] = vertex[tet[i][3]][2] - vertex[tet[i][0]][2]

        for i in range(self.num_elements):
            self.DmInv[i] = self.Dm[i].inverse()
            self.DmInvT[i] = self.DmInv[i].transpose()

    @ti.kernel
    def init_volume(self):
        """初始化四面体体积

        四面体体积: V = |det(Dm)| / 6.0

        :return:
        """

        for i in self.volume:
            self.volume[i] = ti.abs(self.Dm[i].determinant()) / 6.0

    @ti.kernel
    def init_Ta(self):
        """初始化主动张力Ta

        :return:
        """

        # 初始化网格上的Ta
        for i in self.elements:
            self.tet_Ta[i] = 60.0

        # 初始化顶点上的Ta
        for i in self.nodes:
            self.ver_Ta[i] = 600.0

    @ti.kernel
    def set_Ta(self, value: float):
        """设置主动张力Ta的大小

        :param value:
        :return:
        """

        # 设置网格上的Ta值
        for i in self.elements:
            self.tet_Ta[i] = value

        # 设置顶点上的Ta值
        for i in self.nodes:
            self.ver_Ta[i] = value

    @ti.kernel
    def init_nodes_color(self):
        """初始化顶点颜色
        TODO: 更改成根据colormap类初始化顶点颜色

        :return:
        """

        for i in self.nodes_color:
            self.nodes_color[i] = tm.vec3(1.0, 0.5, 0.5)

    @ti.kernel
    def update_color_Vm(self):
        """使用电压更新顶点颜色,
        TODO: 使用colormap类指定顶点颜色的colormap

        :return:
        """

        for i in self.nodes_color:
            # self.nodes_color[i] = tm.vec3([self.Vm[i], 0.0, 1.0 - self.Vm[i]])
            self.nodes_color[i] = colormap.red_bule_linear(self.Vm[i])
