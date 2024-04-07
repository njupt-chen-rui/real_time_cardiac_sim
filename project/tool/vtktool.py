import numpy as np


# 导出vtk格式的四面体模型
# filename: 文件名
# points: numpy格式的顶点数组
# elements: numpy格式的四面体网格数组
# Vm(可选): numpy格式的顶点电压数组
def export_tet_in_vtk(filename, points, elements, Vm=None):
    file = open(filename, "w")
    # vtk格式信息
    file.write("# vtk DataFile Version 2.0\n")
    file.write("vtk from MATLAB\n")
    file.write("ASCII\n")
    file.write("DATASET UNSTRUCTURED_GRID\n")

    # 写入顶点数据
    num_points = np.size(points, 0)
    file.write("POINTS %d double\n" % num_points)
    for i in range(num_points):
        file.write("%f %f %f \n" % (points[i][0], points[i][1], points[i][2]))
    # 写入网格数据
    num_elements = np.size(elements, 0)
    file.write("\nCELLS %d %d\n" % (num_elements, 5 * num_elements))
    for i in range(num_elements):
        file.write("4 %d %d %d %d\n" % (elements[i][0], elements[i][1], elements[i][2], elements[i][3]))
    file.write("\nCELL_TYPES %d \n" % num_elements)
    for _ in range(num_elements):
        file.write("%d\n" % 10)
    # 写入Vm数据
    if Vm:
        file.write("\nPOINT_DATA %d\n" % num_points)
        file.write("SCALARS vm double 1\n")
        file.write("LOOKUP_TABLE default\n")
        num_Vm = np.size(Vm)
        for i in range(num_Vm):
            file.write("%f\n" % Vm[i])
    file.close()
