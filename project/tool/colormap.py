import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Colormap:
    """用于实现colormap的类

    属性:
        type: colormap的类型
    """

    def __init__(self, type_name="red_blue_linear"):
        """初始化colormap类

        :return:
        """
        self.type = type_name

    def set_type(self, type_name):
        """用于在主线程中修改colormap的类型名

        :param type_name: colormap的类型名
        :return:
        """
        if type_name == "cool_to_warm" or type_name == "red_blue_linear":
            self.type = type_name
        else:
            print("Failed! Unavailable colormap type.")

    @ti.func
    def get_rgb(self, input_val: float) -> tm.vec3:
        """将归一化的值映射为rgb颜色, colormap类型由type指定

        :param input_val: 需要映射的值, 需归一化
        :return: rgb表示的颜色值 (tm.vec3)
        """
        val = 0.0
        if input_val < 0.0:
            val = 0.0
        elif input_val > 1.0:
            val = 0.0
        else:
            val = input_val

        res = tm.vec3([0, 0, 0])
        if self.type == "red_blue_linear":
            res = red_blue_linear(val)
        else:
            res = red_blue_linear(val)
        return res


@ti.func
def red_blue_linear(val: float) -> tm.vec3:
    """red_blue_linear 的 colormap 实现

    :param val: [0, 1] 中的某个值
    :return: rgb表示的颜色值 (tm.vec3)
    """
    res_rgb = tm.vec3([0.1229 + val * 0.5978, 0.2254, 0.7207 - val * 0.5978])
    return res_rgb
