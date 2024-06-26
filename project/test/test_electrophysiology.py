"""
测试 electrophysiology.py
"""

import taichi as ti
import taichi.math as tm
import numpy as np
from project.data.whole_heart import meshData
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn


@ti.kernel
def apply_stimulation_test_electrophysiology(body: ti.template(), tag_nid: int, sti_val: float):
    for i in body.nodes:
        dis = (body.nodes[i][0] - body.nodes[tag_nid][0]) * (body.nodes[i][0] - body.nodes[tag_nid][0])\
              + (body.nodes[i][1] - body.nodes[tag_nid][1]) * (body.nodes[i][1] - body.nodes[tag_nid][1])\
              + (body.nodes[i][2] - body.nodes[tag_nid][2]) * (body.nodes[i][2] - body.nodes[tag_nid][2])
        dis = tm.sqrt(dis)
        if dis < 1.0:
            body.Vm[i] = sti_val


def test_electrophysiology():
    """ 测试 Electrophysiology 模块的功能

    :return:
    """

    body, flag_dirichlet, flag_neumman = geo.read_body(meshData)
    electrophysiology_system = elec.Electrophysiology_Aliec_Panfilov(body=body)
    apply_stimulation_test_electrophysiology(body=body, tag_nid=1162, sti_val=1.5)
    electrophysiology_system.sigma_f /= 10.0
    electrophysiology_system.sigma_s /= 10.0
    electrophysiology_system.sigma_n /= 10.0
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_system = dyn.Dynamics_XPBD_SNH_Active(body=body, num_pts_np=num_per_tet_set_np)


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
    camera.position(-0.95338696, 5.68768456, 19.50115459)
    camera.lookat(-0.90405993, 5.36242057, 18.55681875)
    camera.up(0., 1., 0.)

    iter_time = 0
    while window.running:
        if iter_time % 80 == 0:
            apply_stimulation_test_electrophysiology(body=body, tag_nid=1162, sti_val=1.5)
        iter_time += 1
        if electrophysiology_system:
            electrophysiology_system.update(1)
            body.update_color_Vm()
        if dynamics_system:
            dynamics_system.update()

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
    test_electrophysiology()
