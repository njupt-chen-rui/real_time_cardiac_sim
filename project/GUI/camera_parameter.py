import taichi as ti
import taichi.math as tm


@ti.data_oriented
class CameraParameter:
    """相机位置参数预设值

        根据name确定camera的各种参数

    """

    def __init__(self, body_name="unknown"):
        self.camera_pos = tm.vec3(0., 0., 0.)
        self.camera_lookat = tm.vec3(0., 0., 0.)
        self.camera_up = tm.vec3(0., 1., 0.)
        self.camera_fov = 45
        self.name = body_name
        self.init_by_name()

    def init_by_name(self):
        """ 根据物体名称，预设的camera参数

        """

        if self.name == "cube":
            self.camera_pos = tm.vec3(3.41801597, 1.65656349, 3.05081163)
            self.camera_lookat = tm.vec3(2.7179826, 1.31246826, 2.42507068)
            self.camera_up = tm.vec3(0., 1., 0.)
        elif self.name == "whole_heart":
            self.camera_pos = tm.vec3(-0.95338696, 5.68768456, 19.50115459)
            self.camera_lookat = tm.vec3(-0.90405993, 5.36242057, 18.55681875)
            self.camera_up = tm.vec3(0., 1., 0.)
        elif self.name == "lv":
            self.camera_pos = tm.vec3(2.28, 22.6, 34.89)
            self.camera_lookat = tm.vec3(2.24, 22, 34)
            self.camera_up = tm.vec3(0., 1., 0.)

    def set_camera_pos(self, x, y, z):
        self.camera_pos = tm.vec3(x, y, z)

    def set_camera_lookat(self, x, y, z):
        self.camera_lookat = tm.vec3(x, y, z)

    def set_camera_up(self, x, y, z):
        self.camera_up = tm.vec3(x, y, z)
