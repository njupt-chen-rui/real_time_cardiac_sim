import taichi as ti
import taichi.math as tm
import project.Geometry as geo
import project.Dynamics as dyn
from project.GUI.raycaster import Ray, RayCaster


@ti.data_oriented
class Catcher:
    """ 粒子捕捉器

        使用鼠标左键长按时在屏幕上捕捉离其最近的粒子，并选择
    """

    def __init__(self, camera, window, geo_model: geo.Body) -> None:
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

    def catcher(self):
        """ 监测鼠标事件，左键时选中顶点，长按移动时拖拽顶点位置直至松开左键 """

        if self.window.is_pressed(ti.ui.LMB):
            self.start = self.window.get_cursor_pos()
            self.catch_particles(self.start)
            if self.window.get_event(ti.ui.RELEASE):
                self.end = self.window.get_cursor_pos()

        self.start = (-1e5, -1e5)
        self.end = (1e5, 1e5)

    def catch_particle(self, start):
        """ 捕捉距离鼠标选点最近的粒子 """


class Grabber:
    """ 粒子抓取器 """

    def __init__(self, camera, geo_model: geo.Body, dyn_model, resolution) -> None:
        self.camera = camera
        self.resolution = resolution
        self.raycaster = RayCaster(geo_model)
        self.geo_model = geo_model
        self.dyn_model = dyn_model
        self.flag_physicsObject = False
        self.distance = 0.0
        self.prevPos = tm.vec3(0., 0., 0.)
        self.vel = tm.vec3(0., 0., 0.)
        self.time = 0.0

    def increaseTime(self, dt):
        self.time += dt

    def updateRaycaster(self, x, y):
        # TODO:
        self.raycaster.setFromCamera(x, y, self.camera, self.resolution)

    def start(self, x, y):
        if x < 0.25 or x > 0.8 or y < 0.2 or y > 0.9:
            """ hack方法，减少计算量"""
            pass
        else:
            self.updateRaycaster(x, y)
            flag_intersect, dis = self.raycaster.intersect()

            if flag_intersect:
                self.distance = dis
                pos = self.raycaster.ray.get_origin()
                pos += self.raycaster.ray.direction * self.distance
                self.dyn_model.startGrab(pos)
                self.prevPos = tm.vec3(pos[0], pos[1], pos[2])
                self.vel = tm.vec3(0., 0., 0.)
                self.time = 0.0
                self.flag_physicsObject = True

    def move(self, x, y):
        if self.flag_physicsObject:
            self.updateRaycaster(x, y)
            pos = self.raycaster.ray.get_origin()
            pos += self.raycaster.ray.direction * self.distance

            self.vel = tm.vec3(pos[0], pos[1], pos[2])
            self.vel -= self.prevPos
            if self.time > 0.0:
                self.vel /= self.time
            else:
                self.vel = tm.vec3(0., 0., 0.)
            self.prevPos = tm.vec3(pos[0], pos[1], pos[2])
            self.time = 0.0
            self.dyn_model.moveGrabbed(pos, self.vel)

    def end(self):
        if self.flag_physicsObject:
            self.dyn_model.endGrab(self.prevPos, self.vel)
            self.raycaster.clear()
        self.flag_physicsObject = False
