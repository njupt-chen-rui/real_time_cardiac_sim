import numpy as np


def get_surface_from_tet(nodes: np.ndarray, elements: np.ndarray):
    """计算四面体的表面三角形网格，用于可视化

    :param nodes: 四面体顶点
    :param elements: 四面体网格
    :return: 四面体的表面三角网格
    """

    # 表面网格
    surfaces = set()
    # 四面体四个面的顶点索引
    indexes = [(0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)]

    # 遍历所有的四面体网格
    for id_ele, ele in enumerate(elements):
        # 计算当前四面体的中心
        center = np.array([0., 0., 0.])
        for i in range(4):
            center += nodes[ele[i]]
        for i in range(3):
            center[i] /= 4.

        # 当前四面体的四个外表面
        faces = []
        # 判断indexes中每个面的顶点索引代表着正面还是反面，将正面存储在faces中
        for index in indexes:
            (i, j, k) = index
            v0, v1, v2 = nodes[ele[i]], nodes[ele[j]], nodes[ele[k]]
            v0_1 = v1 - v0
            v0_2 = v2 - v0
            vc_0 = v0 - center
            norm = np.cross(v0_1, v0_2)
            sign = np.dot(vc_0, norm)
            if sign > 0:
                faces.append((ele[i], ele[j], ele[k]))
            else:
                faces.append((ele[i], ele[k], ele[j]))

        # 判断face以及其反面是否已经存在于surfaces中了，即判断face是否为四面体网格的内部面
        for face in faces:
            face_inv = (face[0], face[2], face[1])
            if face_inv in surfaces:
                surfaces.remove(face_inv)
            else:
                surfaces.add(face)

    # 将surfaces由set转换为numpy数组
    surfaces = np.array(list(surfaces))
    return surfaces
