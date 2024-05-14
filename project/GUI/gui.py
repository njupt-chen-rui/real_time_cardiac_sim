import taichi as ti
import taichi.math as tm
# from project.Geometry import Body
# from project.GUI import Interaction
import project.Geometry as geo
import project.GUI as gui


# TODO:
# @ti.data_oriented
class Gui:
    """用于可视化的基类

    TODO: 补充注释
    属性:
        iter_time: 迭代时间
        RES: 窗口分辨率
        name: 窗口名称
        is_vsync: 是否开启垂直同步，默认开启
        background_color: 背景颜色，默认白色
        camera_pos: 相机位置
        camera_lookat: camera lookat
        camera_up: camera up vector
    """

    # 仿真迭代次数
    iter_time = 0
    # 相机参数
    # TODO: 相机参数的控制
    camera_pos = tm.vec3(-0.95338696, 5.68768456, 19.50115459)
    camera_lookat = tm.vec3(-0.90405993, 5.36242057, 18.55681875)
    camera_up = tm.vec3(0., 1., 0.)

    def __init__(self, body: geo.Body, electrophysiology_system, dynamics_system, body_name="unknown"):
        self.body = body
        self.electrophysiology_system = electrophysiology_system
        self.dynamics_system = dynamics_system

        # 分辨率
        self.resolution = (1600, 960)
        # 窗口名
        self.window_name = "心脏力电耦合仿真"
        # 是否启用垂直同步
        self.is_vsync = True
        # 背景为白色
        self.background_color = (1., 1., 1.)

        self.interaction_operator = gui.Interaction()
        self.camera_parameter = gui.Camera_parameter(body_name)

    def set_resolution(self, width, height):
        """调整分辨率

        :param width: 宽
        :param height: 高
        :return:
        """
        self.resolution = (width, height)

    def set_name(self, name):
        """ 设置主窗口名称

        :param name: 主窗口名称
        :return:
        """
        self.window_name = name

    def set_body(self, body):
        self.body = body

    def set_elec_sys(self, elec_sys):
        self.elec_sys = elec_sys

    def set_dyn_sys(self, dyn_sys):
        self.dyn_sys = dyn_sys

    def set_background_color(self, r, g, b):
        """ 设置主窗口背景颜色

        :param r: [0, 1]
        :param g: [0, 1]
        :param b: [0, 1]
        :return:
        """
        self.background_color = (r, g, b)

    def set_camera_position(self, x, y, z):
        self.camera_pos = tm.vec3(x, y, z)

    def set_camera_lookat(self, x, y, z):
        self.camera_lookat = tm.vec3(x, y, z)

    def set_camera_up(self, x, y, z):
        self.camera_up = tm.vec3(x, y, z)

    def display(self):
        # 初始化 window, canvas, scene 和 camera 对象
        window = ti.ui.Window(self.window_name, self.resolution, vsync=self.is_vsync)
        canvas = window.get_canvas()
        canvas.set_background_color(self.background_color)
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # 初始化相机位置
        camera.position(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
        camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        camera.up(self.camera_up[0], self.camera_up[1], self.camera_up[2])

        # 设置光源参数
        lengthScale = min(self.resolution[0], 512)
        light_distance = lengthScale / 25.

        is_restart = False
        numSubSteps = 1
        isSolving = False  # 是否运行仿真解算
        is_grab = False  # 是否开始捕捉电刺激施加点
        selector = sel.Selector(camera, window, self.body.nodes, self.body)
        is_apply_ext_force = False  # 是否开始施加外力
        # apply_stimulation1(body=self.body, tag_nid=1162, sti_val=1.5)
        self.body.update_color_Vm()
        use_ap_model = True
        use_fn_model = False
        k = 8.0
        a = 0.01
        b = 0.15
        epsilon_0 = 0.04
        mu_1 = 0.2
        mu_2 = 0.3
        sigma_f = 1.1
        sigma_s = 1.0
        sigma_n = 1.0
        Youngs_modulus = 17000
        Poisson_ratio = 0.45
        kappa = 5.0
        beta = 0.5
        gamma = 1.0
        sigma = 0.0
        dt = 1.0 / 1.29 / 6.0
        numPosIters = 1
        is_save_vtk = False
        flag = True
        # 渲染循环
        while window.running:
            # 交互
            main_gui = window.get_gui()
            with main_gui.sub_window("Controls", 0, 0, 0.25, 0.78) as controls:
                # 仿真是否运行
                isSolving = controls.checkbox("Run", isSolving)
                is_restart = controls.button("Restart")
                dt = controls.slider_float("dt", dt, 0.001, 1.0)
                controls.text("")

                # 电学参数设置
                controls.text("Electrophysiology Model")
                use_ap_model = controls.checkbox("Aliec Panfilov Model", use_ap_model)
                use_fn_model = controls.checkbox("FitzHugh Nagumo Model", use_fn_model)
                k = controls.slider_float("k", k, 1.0, 10.0)
                a = controls.slider_float("a", a, 0.001, 1.0)
                b = controls.slider_float("b", b, 0.001, 1.0)
                beta = controls.slider_float("beta", beta, 0.1, 1.0)
                gamma = controls.slider_float("gamma", gamma, 0.5, 5.0)
                sigma = controls.slider_float("sigma", sigma, 0.0, 1.0)
                epsilon_0 = controls.slider_float("epsilon_0", epsilon_0, 0.01, 1.0)
                mu_1 = controls.slider_float("mu_1", mu_1, 0.01, 1.0)
                mu_2 = controls.slider_float("mu_2", mu_2, 0.01, 1.0)
                sigma_f = controls.slider_float("sigma_f", sigma_f, 0.01, 10.0)
                sigma_s = controls.slider_float("sigma_s", sigma_s, 0.01, 10.0)
                sigma_n = controls.slider_float("sigma_n", sigma_n, 0.01, 10.0)
                # 外界电刺激(捕捉刺激点时暂停仿真)
                is_grab = controls.checkbox("Grab Stimulus Position", is_grab)
                controls.text("")

                # 力学参数设置
                controls.text("Dynamics Model")
                numSubSteps = controls.slider_int("numSubSteps", numSubSteps, 1, 10)
                numPosIters = controls.slider_int("numPosIters", numPosIters, 1, 10)
                Youngs_modulus = controls.slider_float("Young\'s Modulus", Youngs_modulus, 1000.0, 50000.0)
                Poisson_ratio = controls.slider_float("Poisson Ratio", Poisson_ratio, 0.01, 0.4999)
                kappa = controls.slider_float("kappa", kappa, 5.0, 20.0)
                # 外力拖拽(暂停相机视角移动)
                is_apply_ext_force = controls.checkbox("Apply External Force", is_apply_ext_force)
                controls.text("")

                # 保存当前图像
                controls.text("Utility")
                is_save_image = controls.button("Save Image")
                is_save_vtk = controls.button("Save VTK")

            # TODO: 重置仿真
            if is_restart:
                pass

            # 抓取电刺激位置
            if is_grab:
                isSolving = False
                selector.select()
                # 清除选中的电刺激区域
                if window.is_pressed("c"):
                    selector.clear()
                if window.is_pressed("t"):
                    apply_stimulation_gui(self.body, selector, 1.0)
                    self.body.update_color_Vm()
                    # for id in range(self.body.num_nodes):
                    #     if self.body.Vm[id] > 0.0:
                    #         print(id)

            # 施加外力
            if is_apply_ext_force:
                # TODO:
                vid = 3283
                # self.body.Vm[3283] = 1.0
                # self.body.update_color_Vm()
                if flag:
                    self.body.nodes[vid] += tm.vec3([0.5, 0.1, 0.])
                    flag = False
                pass

            # 如果正在抓取电刺激或者施加外力，关闭相机视角旋转
            if not is_grab and not is_apply_ext_force:
                camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)

            # 仿真
            if isSolving:  # 是否开启仿真
                self.iter_time += 1
                self.elec_sys.update(1)
                self.body.update_color_Vm()
                self.dyn_sys.numSubsteps = numSubSteps
                self.dyn_sys.update()

            # 渲染
            # 设置相机
            scene.set_camera(camera)

            # 设置光源
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # 绘制
            scene.mesh(self.body.nodes, indices=self.body.surfaces, two_sided=False,
                       per_vertex_color=self.body.nodes_color)
            # scene.mesh(body.nodes, indices=body.surfaces, two_sided=False, color=(1.0, 0.5, 0.5))

            # 帧显示
            canvas.scene(scene)

            # 保存当前图像
            image_name = "a.jpg"
            if is_save_image:
                window.save_image(image_name)

            window.show()

