'''
@Author: WANG Maonan
@Date: 2024-05-29 17:27:13
@Description: 对 SNIR 进行可视化
@LastEditTime: 2024-05-29 18:45:01
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def render_map(
        x_min, y_min, x_max, y_max, resolution, grid_z,
        trajectories, goal_points, speed, snir_threshold,
        img_path
) -> None:
    fig, ax = plt.subplots()

    start_points = []  # 乘客起点
    end_points = []  # 乘客终点
    for start, end in goal_points:
        start_points.append(start)
        end_points.append(end)

    new_grid_z = generate_new_grid_z(0, 0, x_max, y_max, resolution, grid_z, speed, snir_threshold)

    plot_snir_map(ax, 0, 0, x_max, y_max, snir_threshold, new_grid_z)  # 绘制 SNIR
    # plot_snir_binary(ax, 0, 0, resolution, new_grid_z, snir_min) # 绘制 SNIR 的二值图
    plot_trajectories(ax, trajectories)  # 绘制 aircraft 的轨迹
    plot_goal_points(ax, start_points, color='#1e90ff', marker=">", point_type="S")  # 绘制乘客起点
    plot_goal_points(ax, end_points, color='#1e90ff', marker="o", point_type="D")  # 绘制乘客终点

    # 保存图像
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()


def generate_new_grid_z(x_min, y_min, x_max, y_max, resolution, grid_z, speed, snir_threshold):
    _x_max, _y_max = int((x_max - x_min) // resolution), int((y_max - y_min) // resolution)
    # _x_max, _y_max = grid_z.shape
    # 合并小格作为一个大格
    large_grid = int((speed) / resolution)  # 一个大格里包含多少个小格
    new_grid_z = np.zeros((_x_max, _y_max))

    for x in range(0, _x_max, large_grid):
        for y in range(0, _y_max, large_grid):
            snir = []
            for k in range(x, x + large_grid):
                for l in range(y, y + large_grid):
                    if k < _x_max and l < _y_max:
                        snir.append(grid_z[k, l])
            mean_snir = np.nanmean(snir)
            min_snir = np.nanmin(grid_z)
            if mean_snir < snir_threshold:
                mean_snir = snir_threshold
            if x + large_grid < _x_max and y + large_grid < _y_max:
                new_grid_z[x:x + large_grid, y:y + large_grid] = mean_snir
            elif x + large_grid < _x_max and y + large_grid >= _y_max:
                new_grid_z[x:x + large_grid, y:_y_max] = mean_snir
            elif x + large_grid >= _x_max and y + large_grid < _y_max:
                new_grid_z[x:_x_max, y:y + large_grid] = mean_snir
            else:
                new_grid_z[x:_x_max, y:_y_max] = mean_snir

    return new_grid_z


def plot_snir_binary(ax, x_max, y_max, resolution, grid_z, snir_min):
    """绘制 SNIR 的二值图
    """
    # 计算 x 和 y 坐标
    x_max, y_max = x_max // resolution, y_max // resolution
    threshold = snir_min
    cmap = ListedColormap(['black', 'green'])
    bounds = [-5, threshold, 50]
    norm = BoundaryNorm(bounds, cmap.N)

    # 使用 imshow 显示数据，并设置自定义颜色映射和归一化
    cax = ax.imshow(grid_z.T, extent=(0, x_max * resolution, 0, y_max * resolution), origin='lower', cmap=cmap,
                    norm=norm)

    # 添加颜色条
    cbar = plt.colorbar(cax, ax=ax, ticks=[-5, threshold, 50])
    cbar.ax.set_yticklabels(['-inf', str(threshold), 'inf'])


def plot_snir_map(ax, x_min, y_min, x_max, y_max, threshold, grid_z):
    """绘制 SNIR 的底图
    """
    # 设置 NaN 的颜色为黑色
    grid_z_masked = np.where(grid_z <= threshold, np.nan, grid_z)
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')

    # 绘制 grid_z 数值
    cax = ax.imshow(grid_z_masked, extent=(x_min, x_max, y_min, y_max), origin='lower',cmap=cmap) # 能和sinr原始底图对应
    # 修改颜色低透明度
    cax.set_alpha(0.7)

    # 添加颜色条
    plt.colorbar(cax, ax=ax)


def plot_trajectories(ax, trajectories):
    """绘制车辆轨迹信息
    """
    # first = 0
    for vehicle, path in trajectories.items():
        # 提取 x 和 y 坐标
        x_coords, y_coords = zip(*path)
        # x_coords = -x_coords
        # y_coords = [3260-i for i in y_coords]
        # y_coords = tuple(y_coords)

        # 绘制轨迹的第一个点
        ax.scatter(x_coords[0], y_coords[0], s=100, c='#1e90ff', marker='*')

        # 点的附近加一个标签
        ax.annotate("UAM start", xy=(x_coords[0], y_coords[0]), xytext=(x_coords[0] + 50, y_coords[0] + 50))

        # 绘制轨迹
        ax.plot(x_coords, y_coords, c='#e34a33', label="MSHA-RL")

        # 绘制轨迹的最后一个点
        if path:
            ax.scatter(x_coords[-1], y_coords[-1], s=200, c='#e34a33', marker='*')
            # 点旁边加一个标签
            ax.annotate("UAM current", xy=(x_coords[-1], y_coords[-1]), xytext=(x_coords[-1] - 100, y_coords[-1] - 200))

    # 添加图例
    ax.legend()


def plot_goal_points(ax, goal_points, color, marker, point_type):
    """绘制目标点
    """
    if point_type == "S":
        label = "Passenger Start"
    else:
        label = "Passenger Destination"

    for idx, point in enumerate(goal_points):
        ax.scatter(*point, s=50, c=color, marker=marker, label=f"{point_type}{idx} at {point}")
        # 每一个点旁边加一个标签
        if idx == 1:
            ax.annotate(f"{point_type}{idx+1}", xy=point,
                        xytext=(point[0] - int(150), point[1] - int(150)))

        else:
            ax.annotate(f"{point_type}{idx}", xy=point, xytext=(point[0] + int(50), point[1] + int(50)))
        # ax.annotate(f"{point_type}{idx+1}", xy=point, xytext=(point[0] + int(50), point[1] + int(50)))

    # 添加图例
    # ax.legend()
