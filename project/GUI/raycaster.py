import taichi as ti


@ti.data_oriented
class Ray:
    """ 光线类

    Args:
        origin: 起点
        direction: 方向

    """

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t * self.direction


class RayCaster:
    """ 光线和mesh求交类 """

    def __init__(self):
        self.ray = Ray()

    def setFromCamera(self):
        # TODO:
        pass
