import taichi as ti


# @ti.data_oriented
class Interaction:
    """用于记录交互操作的类
    """

    def __init__(self):
        # 仿真控制
        self.isSolving = False  # 是否运行仿真解算
        self.is_restart = False  # 是否重置仿真
        self.dt = 1.0 / 1.29 / 6.0

        self.iter_time = 0  # 仿真迭代次数

        self.is_save_vtk = False
        self.flag = True

        # TODO: 将电生理模型和动力学模型的dt分开
        self.ele_op = ElectrophysiologyInteraction()
        self.dyn_op = DynamicsInteraction()


class ElectrophysiologyInteraction:
    """ 电生理学仿真交换操作类

    """

    def __init__(self):

        self.ele_model_id = 0
        self.use_ap_model = True  # 默认使用 Aliec Panfilov 模型
        self.use_fn_model = False  # 是否使用 FitzHugh Nagumo 模型

        # 模型共有参数
        self.sigma_f = 1.1
        self.sigma_s = 1.0
        self.sigma_n = 1.0
        self.C_m = 1.0

        self.a = 0.01  # fn: 0.1
        self.epsilon_0 = 0.04  # fn: 0.01

        # ap模型参数
        self.k = 8.0
        self.b = 0.15
        self.mu_1 = 0.2
        self.mu_2 = 0.3

        # fn模型参数
        self.beta = 0.5
        self.gamma = 1.0
        self.sigma = 0.0

        self.is_grab = False  # 是否开始捕捉电刺激施加点
        self.stimulation_value = 20  # 刺激电压值 in [-80, 20]


class DynamicsInteraction:
    """ 动力学仿真交换操作类

    """
    def __init__(self):

        self.numSubSteps = 1
        self.numPosIters = 1

        self.Youngs_modulus = 17000.0
        self.Poisson_ratio = 0.45
        self.kappa = 5.0
        self.is_apply_ext_force = False  # 是否开始施加外力
