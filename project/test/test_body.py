"""
测试 body.py
"""
import project.Geometry.body as geo_body


# if __name__ == "__main__":
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

    Body_ = Body(nodes_np=pos_np,
                 elements_np=tet_np,
                 tet_fiber_np=fiber_tet_np,
                 tet_sheet_np=sheet_tet_np,
                 tet_normal_np=normal_tet_np,
                 num_tet_set_np=num_tet_set_np,
                 tet_set_np=tet_set_np
                 )

    return Body_


def gui(body, elec_sys=None, dyn_sys=None):
    """
    数据可视化
    """

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
            apply_stimulation1(body=body, tag_nid=1162, sti_val=1.5)
        iter_time += 1
        if elec_sys:
            elec_sys.update(1)
            body.update_color_Vm()
        if dyn_sys:
            dyn_sys.update()
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


@ti.kernel
def apply_stimulation1(body: ti.template(), tag_nid: int, sti_val: float):
    for i in body.nodes:
        dis = (body.nodes[i][0] - body.nodes[tag_nid][0]) * (body.nodes[i][0] - body.nodes[tag_nid][0])\
              + (body.nodes[i][1] - body.nodes[tag_nid][1]) * (body.nodes[i][1] - body.nodes[tag_nid][1])\
              + (body.nodes[i][2] - body.nodes[tag_nid][2]) * (body.nodes[i][2] - body.nodes[tag_nid][2])
        dis = tm.sqrt(dis)
        if dis < 1.0:
            body.Vm[i] = sti_val


def example_whole_heart():
    """
    全心实验
    """

    # 从文件中读入数据
    body = read_body()

    # 电仿真模型设置
    # 采用的细胞级模型名称
    electrophysiology_model_name = "Aliec Panfilov"
    if electrophysiology_model_name == "Aliec Panfilov":
        electrophysiology_system = Electrophysiology_Aliec_Panfilov(body=body)
    else:
        # TODO: 别的模型
        electrophysiology_system = Electrophysiology_Aliec_Panfilov(body=body)

    # 设置外界电刺激位置
    # TODO: 刺激电压
    apply_stimulation1(body=body, tag_nid=1162, sti_val=1.5)

    # 设置各向异性扩散参数
    electrophysiology_system.sigma_f /= 10.0
    electrophysiology_system.sigma_s /= 10.0
    electrophysiology_system.sigma_n /= 10.0

    # 动力学仿真模型设置
    # 每个独立集合中包含了多少个四面体
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_system = Dynamics_XPBD_SNH_Active(body=body, num_pts_np=num_per_tet_set_np)

    # 仿真可视化
    gui(body=body, elec_sys=electrophysiology_system, dyn_sys=dynamics_system)

    # 导出vtk
    # import tool.vtktool
    # filename = "../data/res/example4/whole_heart/origin.vtk"
    # np_nodes = body.nodes.to_numpy()
    # np_elements = body.elements.to_numpy()
    # tool.vtktool.write_vtk(filename, np_nodes, np_elements)

    # import tool.vtktool
    # for i in range(20):
    #     if i % 80 == 0:
    #         apply_stimulation1(body=body, tag_nid=1162, sti_val=1.5)
    #     electrophysiology_system.update(1)
    #     dynamics_system.update()
    #     if i % 20 == 0:
    #         filename = "../data/res/example4/whole_heart/%03d.vtk" % i
    #         np_nodes = body.nodes.to_numpy()
    #         np_elements = body.elements.to_numpy()
    #         np_Vm = body.Vm.to_numpy()
    #         for j in range(body.num_nodes):
    #             np_Vm[j] = 100.0 * np_Vm[j] - 80.0
    #         tool.vtktool.write_vtk(filename, np_nodes, np_elements, np_Vm)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    example_whole_heart()
