"""
Visualization Utilities for JSP
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# 尝试解决图表中文字体显示问题 (防止乱码)
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    pass


def plot_train_graph(schedule, n_machines, n_jobs, save_path):
    """
    绘制高铁运行图 (Time-Distance Graph)
    """
    plt.figure(figsize=(24, 12))  # 加大画布以适应更多车站
    n_sections = n_machines // 2
    n_stations = n_sections + 1

    # 动态生成 Y 轴的车站坐标 (0 到 n_sections)
    machine_positions = {}
    for m in range(n_sections):
        # 下行 (Down): 从车站 m 运行到 m+1
        machine_positions[m] = (m, m + 1)
        # 上行 (Up): 对向机器 (n_machines - 1 - m)，从车站 m+1 运行到 m
        opp_m = n_machines - 1 - m
        machine_positions[opp_m] = (m + 1, m)

    # 动态分配颜色
    cmap = plt.cm.get_cmap('tab20', max(20, n_jobs))

    for (job_id, op_idx), (m_id, start_t, end_t) in schedule.items():
        if m_id not in machine_positions:
            continue

        y_start, y_end = machine_positions[m_id]
        color = cmap(job_id % 20)

        # 画运行线段
        plt.plot([start_t, end_t], [y_start, y_end],
                 color=color, linewidth=2.5, marker='o', markersize=4)

        # 标注车次 (只在始发站标注一次，避免图面太乱)
        if op_idx == 0:
            plt.text(start_t, y_start, f' J{job_id}', fontsize=10,
                     color=color, fontweight='bold', va='bottom')

    # 设置 Y 轴刻度和标签 (自适应车站数量)
    plt.yticks(range(n_stations), [f'Station {i}' for i in range(n_stations)])

    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Stations', fontsize=14)
    plt.title('High-Speed Railway Train Graph (运行图)', fontsize=16)

    # 增加细致的网格线
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 高清保存
    plt.close()


def plot_gantt_chart(schedule, n_machines, n_jobs, save_path):
    """
    绘制甘特图 (Gantt Chart)
    """
    plt.figure(figsize=(20, max(10, n_machines * 0.4)))  # 机器越多，画布越高
    cmap = plt.cm.get_cmap('tab20', max(20, n_jobs))

    for (job_id, op_idx), (m_id, start_t, end_t) in schedule.items():
        duration = end_t - start_t
        color = cmap(job_id % 20)

        plt.barh(m_id, duration, left=start_t, color=color, edgecolor='black', alpha=0.8)

        # 在色块正中间写上任务号
        plt.text(start_t + duration / 2, m_id, f'J{job_id}',
                 ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    plt.yticks(range(n_machines), [f'Machine {m}' for m in range(n_machines)])
    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Machines (Sections)', fontsize=14)
    plt.title('Train Scheduling Gantt Chart', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_schedule_to_csv(schedule, save_path):
    """
    保存调度结果为 CSV 文件
    """
    data = []
    for (job_id, op_idx), (m_id, start_t, end_t) in schedule.items():
        data.append({
            'Job_ID': job_id,
            'Operation_Index': op_idx,
            'Machine_ID': m_id,
            'Start_Time': start_t,
            'End_Time': end_t,
            'Duration': end_t - start_t
        })
    df = pd.DataFrame(data)
    # 按车次和执行顺序排序
    df.sort_values(by=['Job_ID', 'Operation_Index'], inplace=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 兼容 Excel 中文