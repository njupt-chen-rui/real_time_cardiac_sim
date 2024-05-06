import taichi as ti
import taichi.math as tm


@ti.data_oriented
class colormap:
    """用于实现colormap的类
    """
    def __init__(self):
        self.type = "red_bule_linear"

    def set_type(self, type_name):
        if type_name == "cool_to_warm" or type_name == "red_bule_linear":
            self.type = type_name

    @ti.func
    def get_rgb(self, input_val: float) -> tm.vec3:
        if input_val < 0.0:
            val = 0.0
        elif input_val > 1.0:
            val = 0.0
        else:
            val = input_val

        if self.type == "red_bule_linear":
            res = red_bule_linear(val)
        else:
            res = red_bule_linear(val)
        return res


@ti.func
def red_bule_linear(val: float) -> tm.vec3:
    res_rgb = tm.vec3([0.1229 + val * 0.5978, 0.2254, 0.7207 - val * 0.5978])
    return res_rgb
