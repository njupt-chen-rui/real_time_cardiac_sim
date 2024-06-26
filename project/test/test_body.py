"""
测试 body.py
"""

import taichi as ti
from project.data.cube import meshData
import project.Geometry as geo


def test_geometry():
    """ 测试Geometry模块的功能

    :return:
    """

    body = geo.read_body(meshData)
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
