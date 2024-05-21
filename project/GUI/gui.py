import taichi as ti
import taichi.math as tm
# from project.GUI import Interaction
import project.Geometry as geo
import project.Electrophysiology as elec
import project.GUI as gui


class Gui:
    """用于可视化的基类

    TODO: 补充注释
    属性:
        iter_time: 迭代时间
        RES: 窗口分辨率
        name: 窗口名称
        is_vsync: 是否开启垂直同步，默认开启
        background_color: 背景颜色，默认白色
        camera_parameter: 相机参数
    """

    def __init__(self, geometry_model: geo.Body, dynamics_model, body_name="unknown"):
        self.geometry_model = geometry_model
        self.electrophysiology_model_ap = elec.Electrophysiology_Aliec_Panfilov(body=geometry_model)
        self.electrophysiology_model_fn = elec.Electrophysiology_FitzHugh_Nagumo(body=geometry_model)
        self.electrophysiology_model = self.electrophysiology_model_ap
        self.dynamics_model = dynamics_model

        # 分辨率
        self.resolution = (1600, 960)
        # 窗口名
        self.window_name = "心脏力电耦合仿真"
        # 是否启用垂直同步
        self.is_vsync = True
        # 背景为白色
        self.background_color = (1., 1., 1.)

        self.interaction_operator = gui.Interaction()
        self.camera_parameter = gui.CameraParameter(body_name)

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

    def set_geometry_model(self, geometry_model):
        """ 设置Body类

        :param geometry_model: 物体几何属性
        :return:
        """
        self.geometry_model = geometry_model

    def set_electrophysiology_model(self, electrophysiology_model):
        """ 设置电生理模型类

        :param electrophysiology_model: 电生理模型类
        :return:
        """
        if self.interaction_operator.ele_op.use_ap_model:
            self.electrophysiology_model_ap = electrophysiology_model
        elif self.interaction_operator.ele_op.use_fn_model:
            self.electrophysiology_model_fn = electrophysiology_model

    def set_dynamics_model(self, dynamics_model):
        """ 设置动力学模型类

        :param dynamics_model: 动力学模型类
        :return:
        """
        self.dynamics_model = dynamics_model

    def set_background_color(self, r, g, b):
        """ 设置主窗口背景颜色

        :param r: [0, 1]
        :param g: [0, 1]
        :param b: [0, 1]
        :return:
        """
        self.background_color = (r, g, b)

    def set_camera_position(self, x, y, z):
        """ 设置 camera position """

        self.camera_parameter.set_camera_pos(x, y, z)

    def set_camera_lookat(self, x, y, z):
        """ 设置 camera lookat """

        self.camera_parameter.set_camera_lookat(x, y, z)

    def set_camera_up(self, x, y, z):
        """ 设置 camera up """

        self.camera_parameter.set_camera_up(x, y, z)

    def display(self):
        """ gui类核心函数

            gui主窗口

        """

        # 初始化 window, canvas, scene 和 camera 对象
        window = ti.ui.Window(self.window_name, self.resolution, vsync=self.is_vsync)
        canvas = window.get_canvas()
        canvas.set_background_color(self.background_color)
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # 初始化相机位置
        camera.position(self.camera_parameter.camera_pos[0],
                        self.camera_parameter.camera_pos[1],
                        self.camera_parameter.camera_pos[2])
        camera.lookat(self.camera_parameter.camera_lookat[0],
                      self.camera_parameter.camera_lookat[1],
                      self.camera_parameter.camera_lookat[2])
        camera.up(self.camera_parameter.camera_up[0],
                  self.camera_parameter.camera_up[1],
                  self.camera_parameter.camera_up[2])

        # 设置光源参数
        lengthScale = min(self.resolution[0], 512)
        light_distance = lengthScale / 25.

        # TODO: body 和 body.nodes不需要重复传值
        selector = gui.Selector(camera, window, self.geometry_model.nodes, self.geometry_model)
        ixop = self.interaction_operator

        # 渲染循环
        while window.running:
            # -------------------------------------------------控制台----------------------------------------------------

            main_gui = window.get_gui()
            with main_gui.sub_window("Controls", 0, 0, 0.25, 0.78) as controls:
                # 仿真控制
                ixop.isSolving = controls.checkbox("Run", ixop.isSolving)
                ixop.is_restart = controls.button("Restart")
                ixop.dt = controls.slider_float("dt", ixop.dt, 0.001, 1.0)
                controls.text("")

                # 电学参数设置
                controls.text("Electrophysiology Model")
                ixop.ele_op.use_ap_model = controls.checkbox("Aliec Panfilov Model", ixop.ele_op.use_ap_model)
                ixop.ele_op.use_fn_model = controls.checkbox("FitzHugh Nagumo Model", ixop.ele_op.use_fn_model)
                # 公有参数
                ixop.ele_op.sigma_f = controls.slider_float("sigma_f", ixop.ele_op.sigma_f, 0.01, 10.0)
                ixop.ele_op.sigma_s = controls.slider_float("sigma_s", ixop.ele_op.sigma_s, 0.01, 10.0)
                ixop.ele_op.sigma_n = controls.slider_float("sigma_n", ixop.ele_op.sigma_n, 0.01, 10.0)
                # TODO: Cm
                ixop.ele_op.a = controls.slider_float("a", ixop.ele_op.a, 0.001, 1.0)
                ixop.ele_op.epsilon_0 = controls.slider_float("epsilon_0", ixop.ele_op.epsilon_0, 0.01, 1.0)
                if ixop.ele_op.use_ap_model:
                    # ap 模型参数
                    ixop.ele_op.k = controls.slider_float("k", ixop.ele_op.k, 1.0, 10.0)
                    ixop.ele_op.b = controls.slider_float("b", ixop.ele_op.b, 0.001, 1.0)
                    ixop.ele_op.mu_1 = controls.slider_float("mu_1", ixop.ele_op.mu_1, 0.01, 1.0)
                    ixop.ele_op.mu_2 = controls.slider_float("mu_2", ixop.ele_op.mu_2, 0.01, 1.0)
                elif ixop.ele_op.use_fn_model:
                    # fn 模型参数
                    ixop.ele_op.beta = controls.slider_float("beta", ixop.ele_op.beta, 0.1, 1.0)
                    ixop.ele_op.gamma = controls.slider_float("gamma", ixop.ele_op.gamma, 0.5, 5.0)
                    ixop.ele_op.sigma = controls.slider_float("sigma", ixop.ele_op.sigma, 0.0, 1.0)
                    controls.text("")
                # 外界电刺激(捕捉刺激点时暂停仿真)
                ixop.ele_op.is_grab = controls.checkbox("Grab Stimulus Position", ixop.ele_op.is_grab)
                ixop.ele_op.stimulation_value = controls.slider_float("Stimulus Voltage", ixop.ele_op.stimulation_value,
                                                                      -80, 20)
                controls.text("")

                # 力学参数设置
                controls.text("Dynamics Model")
                ixop.dyn_op.numSubSteps = controls.slider_int("numSubSteps", ixop.dyn_op.numSubSteps, 1, 10)
                ixop.dyn_op.numPosIters = controls.slider_int("numPosIters", ixop.dyn_op.numPosIters, 1, 10)
                ixop.dyn_op.Youngs_modulus = controls.slider_float("Young\'s Modulus", ixop.dyn_op.Youngs_modulus, 1000.0, 50000.0)
                ixop.dyn_op.Poisson_ratio = controls.slider_float("Poisson Ratio", ixop.dyn_op.Poisson_ratio, 0.01, 0.4999)
                ixop.dyn_op.kappa = controls.slider_float("kappa", ixop.dyn_op.kappa, 5.0, 20.0)
                # TODO: 外力拖拽(暂停相机视角移动)
                # is_apply_ext_force = controls.checkbox("Apply External Force", is_apply_ext_force)
                controls.text("")

                # 保存当前图像
                controls.text("Utility")
                # TODO:
                ixop.is_save_image = controls.button("Save Image")
                ixop.is_save_vtk = controls.button("Save VTK")

            # -------------------------------------------------控制台----------------------------------------------------

            # --------------------------------------------------交互-----------------------------------------------------

            # 如果正在抓取电刺激或者施加外力，关闭相机视角旋转
            if not ixop.ele_op.is_grab and not ixop.dyn_op.is_apply_ext_force:
                camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)

            # 重置仿真
            if ixop.is_restart:
                ixop.isSolving = False
                self.geometry_model.restart()
                self.electrophysiology_model.restart()
                self.dynamics_model.restart()

            # 切换电生理模型
            if ixop.ele_op.use_ap_model and ixop.ele_op.use_fn_model:
                ixop.isSolving = False
                ele_model_id_tmp = 0
                if ixop.ele_op.ele_model_id == 0:
                    ele_model_id_tmp = 1
                    ixop.ele_op.use_ap_model = False
                    ixop.ele_op.use_fn_model = True
                    self.electrophysiology_model = self.electrophysiology_model_fn
                elif ixop.ele_op.ele_model_id == 1:
                    ele_model_id_tmp = 0
                    ixop.ele_op.use_ap_model = True
                    ixop.ele_op.use_fn_model = False
                    self.electrophysiology_model = self.electrophysiology_model_ap
                ixop.ele_op.ele_model_id = ele_model_id_tmp
                self.geometry_model.restart()
                self.electrophysiology_model.restart()
                self.dynamics_model.restart()

            # 设置电生理参数
            self.electrophysiology_model.sigma_f = ixop.ele_op.sigma_f
            self.electrophysiology_model.sigma_s = ixop.ele_op.sigma_s
            self.electrophysiology_model.sigma_n = ixop.ele_op.sigma_n
            if ixop.ele_op.use_ap_model:
                self.electrophysiology_model.a = ixop.ele_op.a
                self.electrophysiology_model.epsilon_0 = ixop.ele_op.epsilon_0
                self.electrophysiology_model.k = ixop.ele_op.k
                self.electrophysiology_model.b = ixop.ele_op.b
                self.electrophysiology_model.mu_1 = ixop.ele_op.mu_1
                self.electrophysiology_model.mu_2 = ixop.ele_op.mu_2
            elif ixop.ele_op.use_fn_model:
                self.electrophysiology_model.a = ixop.ele_op.a
                self.electrophysiology_model.epsilon_0 = ixop.ele_op.epsilon_0
                self.electrophysiology_model.beta = ixop.ele_op.beta
                self.electrophysiology_model.gamma = ixop.ele_op.gamma
                self.electrophysiology_model.sigma = ixop.ele_op.sigma

            # TODO: 抓取电刺激位置
            if ixop.ele_op.is_grab:
                ixop.isSolving = False
                selector.select()
                # 清除选中的电刺激区域
                if window.is_pressed("c"):
                    selector.clear()
                # 确认已选中的电刺激区域
                if window.is_pressed("t"):
                    apply_stimulation_with_selector(self.geometry_model, selector,
                                                    (ixop.ele_op.stimulation_value + 80) / 100.0)
                    self.geometry_model.update_color_Vm()

            # 设计力学模型参数
            self.dynamics_model.numSubsteps = ixop.dyn_op.numSubSteps
            self.dynamics_model.numPosIters = ixop.dyn_op.numPosIters
            self.dynamics_model.Youngs_modulus = ixop.dyn_op.Youngs_modulus
            self.dynamics_model.Poisson_ratio = ixop.dyn_op.Poisson_ratio
            self.dynamics_model.kappa = ixop.dyn_op.kappa

            # TODO: 施加外力
            if ixop.dyn_op.is_apply_ext_force:
                ixop.isSolving = False

            # --------------------------------------------------交互-----------------------------------------------------

            # --------------------------------------------------仿真-----------------------------------------------------

            if ixop.isSolving:  # 是否开启仿真
                ixop.iter_time += 1
                self.electrophysiology_model.update(1)
                self.geometry_model.update_color_Vm()
                self.dynamics_model.numSubsteps = ixop.dyn_op.numSubSteps
                self.dynamics_model.update()

            # --------------------------------------------------仿真-----------------------------------------------------

            # --------------------------------------------------渲染-----------------------------------------------------

            # 设置相机
            scene.set_camera(camera)

            # 设置光源
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # 绘制
            scene.mesh(self.geometry_model.nodes, indices=self.geometry_model.surfaces, two_sided=False,
                       per_vertex_color=self.geometry_model.nodes_color)

            # 帧显示
            canvas.scene(scene)

            # # 保存当前图像
            # image_name = "a.jpg"
            # if is_save_image:
            #     window.save_image(image_name)

            # 显示
            window.show()

            # --------------------------------------------------渲染-----------------------------------------------------


@ti.kernel
def apply_stimulation_with_selector(body: ti.template(), sec: ti.template(), sti_val: float):
    for i in body.nodes:
        if sec.is_in_rect[i]:
            body.Vm[i] = sti_val
