import taichi as ti
# from project.Geometry import Body
import project.Geometry as geo


@ti.data_oriented
class Selector:
    """ 粒子选择器

        使用鼠标左键在屏幕上绘制一个矩形，然后选择矩形中的粒子
    """

    def __init__(self, camera, window, particle_pos, geo_model: geo.Body) -> None:
        """
        Args:

            camera (ti.ui.Camera): the camera used in the scene
            window (ti.ui.Window): the window used in the scene
            particle_pos (3d ti.Vector.field): the position of all the particles which could be selected
        """

        self.camera = camera
        self.window = window
        self.canvas = window.get_canvas()
        w, h = window.get_window_shape()
        self.aspect = w / h
        self.start = (-1e5, -1e5)
        self.end = (1e5, 1e5)

        self.geo_model = geo_model
        # 每个顶点的颜色
        self.per_vertex_color = geo_model.nodes_color
        # 屏幕上顶点二维位置
        self.screen_pos = ti.Vector.field(2, shape=geo_model.num_nodes, dtype=float)
        # 顶点是否在框选的二维矩形中
        self.is_in_rect = ti.field(dtype=int, shape=geo_model.num_nodes)
        # 矩形的4条边(8个顶点，用于显示框选矩形)
        self.rect_verts = ti.Vector.field(2, dtype=ti.f32, shape=8)
        # 顶点的三维坐标
        self.particle_pos = geo_model.nodes
        # 被选中顶点的数量
        self.num_selected = ti.field(dtype=int, shape=())
        # 被选中顶点的id？
        self.selected_ids = ti.field(shape=geo_model.num_nodes, dtype=int)
        self.selected_ids.fill(-1)

    def select_particles(self, start, end):
        """ 选中顶点 """

        # 顶点世界坐标
        world_pos = self.particle_pos
        # 左下
        leftbottom = [min(start[0], end[0]), min(start[1], end[1])]
        # 右上
        righttop = [max(start[0], end[0]), max(start[1], end[1])]
        # view matrix
        view_ti = ti.math.mat4(self.camera.get_view_matrix())
        # projection matrix
        proj_ti = ti.math.mat4(self.camera.get_projection_matrix(self.aspect))

        @ti.kernel
        def world_to_screen_kernel(world_pos: ti.template()):
            for i in range(world_pos.shape[0]):
                pos_homo = ti.math.vec4([world_pos[i][0], world_pos[i][1], world_pos[i][2], 1.0])
                ndc = pos_homo @ view_ti @ proj_ti  # CAUTION: right multiply
                ndc /= ndc[3]

                self.screen_pos[i][0] = ndc[0]
                self.screen_pos[i][1] = ndc[1]
                # from [-1,1] scale to [0,1]
                self.screen_pos[i][0] = (self.screen_pos[i][0] + 1) / 2
                self.screen_pos[i][1] = (self.screen_pos[i][1] + 1) / 2

                if leftbottom[0] < self.screen_pos[i][0] < righttop[0] and \
                        leftbottom[1] < self.screen_pos[i][1] < righttop[1]:
                    self.is_in_rect[i] = True
                    self.per_vertex_color[i] = [1.0, 1.0, 0.0]

        world_to_screen_kernel(world_pos)

    def select(self):
        """ 监测鼠标事件，画出框选长方体，被选中其中的顶点 """

        if self.window.is_pressed(ti.ui.LMB):
            self.start = self.window.get_cursor_pos()
            if self.window.get_event(ti.ui.RELEASE):
                self.end = self.window.get_cursor_pos()
            self.rect(self.start[0], self.start[1], self.end[0], self.end[1])
            # 画框选的长方体
            self.canvas.lines(vertices=self.rect_verts, color=(1.0, 1.0, 0.0), width=0.005)
            # 选点
            self.select_particles(self.start, self.end)

    def clear(self):
        """ 清楚所有顶点的选中 """
        self.geo_model.update_color_Vm()
        self.is_in_rect.fill(0)

    def rect(self, x_min, y_min, x_max, y_max):
        """ 用于画出框选长方体 """
        self.rect_verts[0] = [x_min, y_min]
        self.rect_verts[1] = [x_max, y_min]
        self.rect_verts[2] = [x_min, y_max]
        self.rect_verts[3] = [x_max, y_max]
        self.rect_verts[4] = [x_min, y_min]
        self.rect_verts[5] = [x_min, y_max]
        self.rect_verts[6] = [x_max, y_min]
        self.rect_verts[7] = [x_max, y_max]


def visualize(particle_pos):
    window = ti.ui.Window("visualizer", (1080, 720), vsync=True)
    camera = ti.ui.Camera()
    camera.position(0, 0, 0)
    camera.lookat(0, 0, -1)
    camera.fov(45)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()

    selector = Selector(camera, window, particle_pos)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        selector.select()
        if window.is_pressed("c"):
            selector.clear()
        if window.is_pressed("i"):
            print(selector.get_ids())
        scene.particles(particle_pos, radius=0.01, per_vertex_color=selector.per_vertex_color)

        canvas.scene(scene)
        window.show()
        # ti.profiler.print_kernel_profiler_info()
