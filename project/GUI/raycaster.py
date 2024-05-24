import taichi as ti
import taichi.math as tm
import project.Geometry as geo

PI = 3.14159265


@ti.data_oriented
class Ray:
    """ 光线类

    Args:
        origin: 起点
        direction: 方向

    """

    def __init__(self, origin: tm.vec3, direction: tm.vec3):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t * self.direction

    def get_origin(self):
        return tm.vec3(self.origin[0], self.origin[1], self.origin[2])

    def get_direction(self):
        return tm.vec3(self.direction[0], self.direction[1], self.direction[2])


@ti.data_oriented
class RayCaster:
    """ 光线和mesh求交类 """

    def __init__(self, geo_model: geo.Body) -> None:
        self.origin = tm.vec3(0, 0, 0)
        self.direction = tm.vec3(0, 0, 0)
        self.ray = Ray(self.origin, self.direction)
        self.geo_model = geo_model
        self.num_surfaces = self.geo_model.surfaces.shape[0] // 3

        self.t = ti.field(ti.f32, shape=(self.num_surfaces,))
        self.max_val = 100000000.0

    @ti.kernel
    def clear(self):
        for i in self.t:
            self.t[i] = self.max_val

    def setFromCamera(self, x, y, camera: ti.ui.Camera, resolution):
        # TODO:
        lookfrom = camera.curr_position
        lookat = camera.curr_lookat
        vup = camera.curr_up
        fov = 45
        aspect_ratio = resolution[0] / resolution[1]
        theta = fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = aspect_ratio * half_height
        cam_origin = tm.vec3(lookfrom[0], lookfrom[1], lookfrom[2])
        w = (lookfrom - lookat).normalized()
        u = (vup.cross(w)).normalized()
        v = w.cross(u)
        # cam_lower_left_corner = tm.vec3(-half_width, -half_height, -1.0)
        cam_lower_left_corner = cam_origin - half_width * u - half_height * v - w
        cam_horizontal = 2 * half_width * u
        cam_vertical = 2 * half_height * v

        self.ray = Ray(cam_origin, cam_lower_left_corner + x * cam_horizontal + y * cam_vertical - cam_origin)
        self.origin = self.ray.get_origin()
        self.direction = self.ray.get_direction()
        # print(self.origin, self.direction)

    def intersect(self) -> (bool, float):
        self.get_intersect()
        # self.get_intersect_cpu()
        min_dic = self.max_val
        flag = False
        for i in range(self.num_surfaces):
            if self.t[i] < min_dic:
                min_dic = self.t[i]
                flag = True

        return flag, min_dic

    @ti.kernel
    def get_intersect(self):
        O = self.origin
        d = self.direction
        surface, nodes = ti.static(self.geo_model.surfaces, self.geo_model.nodes)
        for i in self.t:
            vid0, vid1, vid2 = surface[3 * i + 0], surface[3 * i + 1], surface[3 * i + 2]
            p0, p1, p2 = nodes[vid0], nodes[vid1], nodes[vid2]
            p1_0 = p1 - p0
            p2_0 = p2 - p0
            N = tm.cross(p1_0, p2_0)
            molecule = tm.dot((O - p0), N)
            denominator = tm.dot(d, N)
            if ti.abs(denominator) > 1e-6:
                self.t[i] = -1.0 * molecule / denominator
                P = O + self.t[i] * d
                S_ABC = 0.5 * vec_length(N)
                S_PAB = triangle_area(P, p0, p1)
                S_PBC = triangle_area(P, p1, p2)
                S_PCA = triangle_area(P, p2, p0)
                if ti.abs(S_ABC - S_PAB - S_PBC - S_PCA) < 1e-6:
                    self.t[i] = self.max_val
                if self.t[i] <= 0:
                    self.t[i] = self.max_val
            else:
                self.t[i] = self.max_val

    def get_intersect_cpu(self):
        O = self.origin
        d = self.direction
        surface, nodes = ti.static(self.geo_model.surfaces, self.geo_model.nodes)
        for i in range(self.num_surfaces):
            vid0, vid1, vid2 = surface[3 * i + 0], surface[3 * i + 1], surface[3 * i + 2]
            p0, p1, p2 = nodes[vid0], nodes[vid1], nodes[vid2]
            p1_0 = p1 - p0
            p2_0 = p2 - p0
            N = my_cross(p1_0, p2_0)
            molecule = my_dot((O - p0), N)
            denominator = my_dot(d, N)
            if ti.abs(denominator) > 1e-6:
                self.t[i] = molecule / denominator
                P = O + self.t[i] * d
                S_ABC = 0.5 * my_vec_length(N)
                S_PAB = my_triangle_area(P, p0, p1)
                S_PBC = my_triangle_area(P, p1, p2)
                S_PCA = my_triangle_area(P, p2, p0)
                if S_ABC != S_PAB + S_PBC + S_PCA:
                    self.t[i] = self.max_val
            else:
                self.t[i] = self.max_val


@ti.func
def triangle_area(a: tm.vec3, b: tm.vec3, c: tm.vec3) -> float:
    v1 = b - a
    v2 = c - a
    return vec_length(tm.cross(v1, v2)) * 0.5


@ti.func
def vec_length(vec: tm.vec3) -> float:
    return tm.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def my_cross(v1: tm.vec3, v2: tm.vec3) -> tm.vec3:
    return tm.vec3(v1[1] * v2[2] - v1[2] * v2[1],
                   v1[2] * v2[0] - v1[0] * v2[2],
                   v1[0] * v2[1] - v1[1] * v2[0])


def my_dot(v1: tm.vec3, v2: tm.vec3) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def my_vec_length(vec: tm.vec3) -> float:
    return ti.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def my_triangle_area(a: tm.vec3, b: tm.vec3, c: tm.vec3) -> float:
    v1 = b - a
    v2 = c - a
    return my_vec_length(my_cross(v1, v2)) * 0.5

