# cube:
#     camera.position(3.41801597, 1.65656349, 3.05081163)
#     camera.lookat(2.7179826, 1.31246826, 2.42507068)
#     camera.up(0., 1., 0.)

# whole heart
#     camera.position(-0.95338696, 5.68768456, 19.50115459)
#     camera.lookat(-0.90405993, 5.36242057, 18.55681875)
#     camera.up(0., 1., 0.)

import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Camera_parameter:
    """相机位置参数预设值"""

    def __init__(self, body_name="cube"):
        self.name = body_name

    def get_camera_position(self):
        res = (0., 0., 0.)
        if self.name == "cube":
            res = (3.41801597, 1.65656349, 3.05081163)
        elif self.name == "whole_heart":
            res = (-0.95338696, 5.68768456, 19.50115459)
        return res

    def get_camera_lookat(self):
        res = (0., 0., 0.)
        if self.name == "cube":
            res = (2.7179826, 1.31246826, 2.42507068)
        elif self.name == "whole_heart":
            res = (-0.90405993, 5.36242057, 18.55681875)
        return res

    def get_camera_up(self):
        res = (0., 1., 0.)
        if self.name == "cube":
            res = (0., 1., 0.)
        elif self.name == "whole_heart":
            res = (0., 1., 0.)
        return res


# test
# if __name__ == "__main__":
#     camera_para = Camera_parameter()
#     res = camera_para.get_camera_position()
#     print(res[0])
