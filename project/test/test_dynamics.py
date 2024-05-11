"""
测试 dynamics.py
"""

import taichi as ti
import taichi.math as tm
import numpy as np
from project.data.cube import meshData
import project.Geometry as geo
import project.Dynamics as dyn


@ti.kernel
def init_Vm_linear(body: ti.template()):
    for i in body.Vm:
        body.Vm[i] = body.nodes[i][1] * 30.0
        body.ver_Ta[i] = 0.5 * body.Vm[i]

    for i in body.elements:
        id0, id1, id2, id3 = body.elements[i][0], body.elements[i][1], body.elements[i][2], body.elements[i][3]
        body.tet_Ta[i] = (body.ver_Ta[id0] + body.ver_Ta[id1] + body.ver_Ta[id2] + body.ver_Ta[id3]) / 4.0


@ti.kernel
def init_dirichlet_bou(body: ti.template()):
    for i in body.nodes:
        if abs(body.nodes[i][1]) < 1e-12:
            body.bou_tag_dirichlet[i] = 1


@ti.kernel
def init_fiber(body: ti.template()):
    for i in body.tet_fiber:
        body.tet_fiber[i] = tm.vec3(0.0, 1.0, 0.0)
        body.tet_sheet[i] = tm.vec3(1.0, 0.0, 0.0)


def test_dynamics():
    """ 测试 Dynamics 模块的功能

    :return:
    """

    body, flag_dirichlet, flag_neumann = geo.read_body(meshData)

    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_sys = dyn.Dynamics_XPBD_SNH_Active(body=body, num_pts_np=num_per_tet_set_np,
                                                tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    # 施加线性分布的电压
    init_Vm_linear(body=body)
    init_dirichlet_bou(body=body)
    init_fiber(body=body)
    dynamics_sys.dt = 1.0 / 50.0
    dynamics_sys.flag_update_Ta = False
    # dynamics_sys.numPosIters = 10
    body.update_color_Vm()

    # 设置窗口参数
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    # init the window, canvas, scene and camera
    window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1., 1., 1.))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initial camera position
    camera.position(3.41801597, 1.65656349, 3.05081163)
    camera.lookat(2.7179826, 1.31246826, 2.42507068)
    camera.up(0., 1., 0.)

    iter_time = 0
    while window.running:
        dynamics_sys.update()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.ambient_light(color=(0.5, 0.5, 0.5))

        # draw
        scene.mesh(body.nodes, indices=body.surfaces, two_sided=False, per_vertex_color=body.nodes_color)
        # scene.mesh(body.nodes, indices=body.surfaces, two_sided=False, color=(1.0, 0.5, 0.5))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    test_dynamics()
