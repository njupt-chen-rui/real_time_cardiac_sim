"""
测试 body.py
"""

import taichi as ti
import numpy as np
import project.Geometry.body as geo_body
import project.tool.colormap
from project.data.cube import meshData


def read_body():
    """
    从文件中读入数据并创建Body对象
    """
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))
    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # tet_normal方向
    normal_tet_np = np.array(meshData['normalDirection'], dtype=float)
    normal_tet_np = normal_tet_np.reshape((-1, 3))
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # colormap
    test_colormap = project.tool.colormap.Colormap()

    Body_ = geo_body.Body(
                 colormap=test_colormap,
                 nodes_np=pos_np,
                 elements_np=tet_np,
                 tet_fiber_np=fiber_tet_np,
                 tet_sheet_np=sheet_tet_np,
                 tet_normal_np=normal_tet_np,
                 num_tet_set_np=num_tet_set_np,
                 tet_set_np=tet_set_np
                 )

    return Body_


def test_geometry():
    """ 测试Geometry模块的功能

    :return:
    """

    body = read_body()
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

    while window.running:

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
    test_geometry()
