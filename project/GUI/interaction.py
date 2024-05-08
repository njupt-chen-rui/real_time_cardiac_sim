import taichi as ti


@ti.data_oriented
class Interaction:
    """用于记录交互操作的类
    """

    def __init__(self):
        self.isSolving = False  # 是否运行仿真解算
        self.is_grab = False  # 是否开始捕捉电刺激施加点
