import taichi as ti
import taichi.math as tm

if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    resolution = (1600, 960)
    background_color = (1., 1., 1.)
    window = ti.ui.Window("test", resolution)
    canvas = window.get_canvas()
    canvas.set_background_color(background_color)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # 初始化相机位置
    camera.position(0.0, 1.0, -5.0)
    camera.lookat(0.0, 1.0, -1.0)
    camera.up(0., 1., 0.)

    # 设置光源参数
    lengthScale = min(resolution[0], 512)
    light_distance = lengthScale / 25.
    old_start = (-1, -1)
    while window.running:
        # 设置相机
        scene.set_camera(camera)

        # 设置光源
        scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.ambient_light(color=(0.5, 0.5, 0.5))
        if window.is_pressed(ti.ui.LMB):
            start = window.get_cursor_pos()
            if start != old_start:
                old_start = start
                print(start)
        # 帧显示
        canvas.scene(scene)

        window.show()
