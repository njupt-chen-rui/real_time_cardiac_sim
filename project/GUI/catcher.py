import taichi as ti
import taichi.math as tm
import project.Geometry as geo
import project.Dynamics as dyn


@ti.data_oriented
class Catcher:
    """ 粒子捕捉器

        使用鼠标左键长按时在屏幕上捕捉离其最近的粒子，并选择
    """

    def __init__(self, camera: ti.ui.Camera, window: ti.ui.Window, geo_model: geo.Body) -> None:
        """"""

        self.camera = camera
        self.window = window
        self.canvas = window.get_canvas()
        w, h = window.get_window_shape()
        self.aspect = w / h
        self.start = (-1e5, -1e5)
        self.end = (1e5, 1e5)

        # 物体的几何属性
        self.geo_model = geo_model
        # 屏幕上顶点的二维坐标
        self.screen_pos = ti.Vector.field(2, shape=geo_model.num_nodes, dtype=float)
        # 顶点的三维坐标
        self.particle_pos = geo_model.nodes
        self.is_catched = ti.field(dtype=int, shape=geo_model.num_nodes)
        # 框选半径
        self.radius = 1e-2
        self.catch_id = -1
        self.mouse_down = False
        self.t = 0
        self.num_catched = 0

    def catcher(self):
        """ 监测鼠标事件，左键时选中顶点，长按移动时拖拽顶点位置直至松开左键 """

        # if self.window.is_pressed(ti.ui.LMB):
        #     self.start = self.window.get_cursor_pos()
        #     if self.window.get_event(ti.ui.RELEASE):
        #         self.end = self.window.get_cursor_pos()
        #     self.catch_particle_old(self.start, self.end)

        if self.window.is_pressed(ti.ui.LMB):
            mouse_pos = self.window.get_cursor_pos()
            if mouse_pos[0] < 0.25 or mouse_pos[0] > 0.8 or mouse_pos[1] < 0.2 or mouse_pos[1] > 0.9:
                """ hack方法，减少计算量"""
                pass
            elif self.mouse_down:
                self.move_particle(mouse_pos)
            else:
                self.catch_particle(mouse_pos)
                self.mouse_down = True
        else:
            self.mouse_down = False


    def catch_particle_old(self, start_coord, end_coord):
        """ 捕捉距离鼠标选点最近的粒子 """
        self.clear()
        # 顶点世界坐标
        world_pos = self.particle_pos
        # view matrix
        view_ti = ti.math.mat4(self.camera.get_view_matrix())
        # projection matrix
        proj_ti = ti.math.mat4(self.camera.get_projection_matrix(self.aspect))

        @ti.kernel
        def world_to_screen_kernel(world_pos: ti.template()):
            for i in range(world_pos.shape[0]):
                pos_homo = ti.math.vec4([world_pos[i][0], world_pos[i][1], world_pos[i][2], 1.0])
                ndc = pos_homo @ view_ti @ proj_ti
                ndc /= ndc[3]

                self.screen_pos[i][0] = ndc[0]
                self.screen_pos[i][1] = ndc[1]
                self.screen_pos[i][0] = (self.screen_pos[i][0] + 1.0) * 0.5
                self.screen_pos[i][1] = (self.screen_pos[i][1] + 1.0) * 0.5

                if (start_coord[0] - self.radius < self.screen_pos[i][0] < start_coord[0] + self.radius) and (
                        start_coord[1] - self.radius < self.screen_pos[i][1] < start_coord[1] + self.radius):
                    self.is_catched[i] = 1

        world_to_screen_kernel(world_pos)

        origin = self.camera.curr_position
        min_dict = 1000000.0
        num_catched = 0
        for i in range(self.geo_model.num_nodes):
            if self.is_catched[i]:
                num_catched += 1
                dict = (self.geo_model.nodes[i][0] - origin[0]) * (self.geo_model.nodes[i][0] - origin[0]) + (
                        self.geo_model.nodes[i][1] - origin[1]) * (self.geo_model.nodes[i][1] - origin[1]) + (
                        self.geo_model.nodes[i][2] - origin[2]) * (self.geo_model.nodes[i][2] - origin[2])
                if dict < min_dict:
                    min_dict = dict
                    self.catch_id = i

        if num_catched > 0:
            start_direction, t = self.intersection(self.catch_id, start_coord)
            # P = origin + t * start_direction
            # self.geo_model.nodes[self.catch_id] = P
            end_direction = self.get_direction(end_coord[0], end_coord[1])
            P = origin + t * end_direction
            self.geo_model.nodes[self.catch_id] = P


            # 调试输出
            # self.geo_model.Vm[self.catch_id] = 1.0
            # self.geo_model.update_color_Vm()

    def intersection(self, vid, coord):
        O = self.camera.curr_position
        d = self.get_direction(coord[0], coord[1])
        pos = self.geo_model.nodes[vid]
        a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        b = 2.0 * (d[0] * (O[0] - pos[0]) + d[1] * (O[1] - pos[1]) + d[2] * (O[2] - pos[2]))
        # c = (O[0] - pos[0]) * (O[0] - pos[0]) + (O[1] - pos[1]) * (O[1] - pos[1]) + (O[2] - pos[2]) * (O[2] - pos[2])
        t = -0.5 * b / a
        return d, t

    def get_direction(self, x, y):
        lookfrom = self.camera.curr_position
        lookat = self.camera.curr_lookat
        vup = self.camera.curr_up
        fov = 45
        aspect_ratio = self.aspect
        PI = 3.14159265
        theta = fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = aspect_ratio * half_height
        cam_origin = tm.vec3(lookfrom[0], lookfrom[1], lookfrom[2])
        w = (lookfrom - lookat).normalized()
        u = (vup.cross(w)).normalized()
        v = w.cross(u)
        cam_lower_left_corner = cam_origin - half_width * u - half_height * v - w
        cam_horizontal = 2 * half_width * u
        cam_vertical = 2 * half_height * v
        return cam_lower_left_corner + x * cam_horizontal + y * cam_vertical - cam_origin

    @ti.kernel
    def clear(self):
        for i in self.is_catched:
            self.is_catched[i] = 0

    def catch_particle(self, start_coord):
        """ 捕捉距离鼠标选点最近的粒子 """
        self.clear()
        # 顶点世界坐标
        world_pos = self.particle_pos
        # view matrix
        view_ti = ti.math.mat4(self.camera.get_view_matrix())
        # projection matrix
        proj_ti = ti.math.mat4(self.camera.get_projection_matrix(self.aspect))

        @ti.kernel
        def world_to_screen_kernel(world_pos: ti.template()):
            for i in range(world_pos.shape[0]):
                pos_homo = ti.math.vec4([world_pos[i][0], world_pos[i][1], world_pos[i][2], 1.0])
                ndc = pos_homo @ view_ti @ proj_ti
                ndc /= ndc[3]

                self.screen_pos[i][0] = ndc[0]
                self.screen_pos[i][1] = ndc[1]
                self.screen_pos[i][0] = (self.screen_pos[i][0] + 1.0) * 0.5
                self.screen_pos[i][1] = (self.screen_pos[i][1] + 1.0) * 0.5

                if (start_coord[0] - self.radius < self.screen_pos[i][0] < start_coord[0] + self.radius) and (
                        start_coord[1] - self.radius < self.screen_pos[i][1] < start_coord[1] + self.radius):
                    self.is_catched[i] = 1

        world_to_screen_kernel(world_pos)

        origin = self.camera.curr_position
        min_dict = 1000000.0
        self.num_catched = 0
        for i in range(self.geo_model.num_nodes):
            if self.is_catched[i]:
                self.num_catched += 1
                dict = (self.geo_model.nodes[i][0] - origin[0]) * (self.geo_model.nodes[i][0] - origin[0]) + (
                        self.geo_model.nodes[i][1] - origin[1]) * (self.geo_model.nodes[i][1] - origin[1]) + (
                        self.geo_model.nodes[i][2] - origin[2]) * (self.geo_model.nodes[i][2] - origin[2])
                if dict < min_dict:
                    min_dict = dict
                    self.catch_id = i

        if self.num_catched > 0:
            start_direction, self.t = self.intersection(self.catch_id, start_coord)
            P = origin + self.t * start_direction
            self.geo_model.nodes[self.catch_id] = P

            # 调试输出
            # self.geo_model.Vm[self.catch_id] = 1.0
            # self.geo_model.update_color_Vm()

    def move_particle(self, end_coord):
        """ 移动鼠标用于操作粒子 """
        if self.num_catched > 0:
            origin = self.camera.curr_position
            end_direction = self.get_direction(end_coord[0], end_coord[1])
            P = origin + self.t * end_direction
            self.geo_model.nodes[self.catch_id] = P

        # 调试输出
        # self.geo_model.Vm[self.catch_id] = 1.0
        # self.geo_model.update_color_Vm()
