
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_train_graph(schedule, n_machines, n_jobs, save_path='train_graph.png'):
    """
    绘制列车运行图（Train Graph）
    X轴：时间
    Y轴：车站/区间（Machine）
    线条：列车运行轨迹
    """
    plt.figure(figsize=(15, 8))
    
    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, n_jobs))
    
    # 绘制每个列车的运行线
    for job_id in range(n_jobs):
        # 获取该列车的所有工序调度信息
        job_ops = []
        for (j, op), (m, start, end) in schedule.items():
            if j == job_id:
                job_ops.append((op, m, start, end))
        
        # 按工序顺序排序
        job_ops.sort(key=lambda x: x[0])
        
        # 提取时间和位置点
        times = []
        positions = []
        
        for op, m, start, end in job_ops:
            # 假设Machine ID对应物理位置顺序（0 -> n_machines-1）
            # 起点
            times.append(start)
            positions.append(m)
            # 终点
            times.append(end)
            positions.append(m + 1) # 修正：对于运行图，终点应该是下一个车站的位置（Machine ID + 1）
            # 注意：这里的 Machine ID 代表的是区间。例如 Machine 0 是 A-B。
            # 起点是 A (位置 0)，终点是 B (位置 1)。
            # 所以 Position = Machine ID (起点) -> Machine ID + 1 (终点)

        # 绘制折线（表示列车移动）
        if job_id == 4: # 天窗列车 (Train 4)
            # 使用阴影区域表示天窗
            # 由于可能跨越多个区间，我们需要分段绘制
            for i in range(0, len(times), 2):
                t_start = times[i]
                t_end = times[i+1]
                p_start = positions[i]
                p_end = positions[i+1]
                
                # 绘制阴影矩形 (T-D Graph中通常是一个平行四边形区域)
                # 这里我们画一个覆盖该时间段和区间的矩形块
                # plt.fill_betweenx([p_start, p_end], t_start, t_end, color='gray', alpha=0.3)
                
                # 或者直接画一条很宽的线
                plt.plot([t_start, t_end], [p_start, p_end], '-', color='gray', linewidth=10, alpha=0.3, label='Skylight (T4)' if i==0 else "")
                
                # 同时也画一条细线以便看清中心
                plt.plot([t_start, t_end], [p_start, p_end], '--', color='black', linewidth=1)
        else:
            # 普通列车
            plt.plot(times, positions, 'o-', label=f'Train {job_id}', color=colors[job_id], linewidth=2, markersize=4)
            # 标注车次
            if times:
                plt.text(times[0], positions[0], f'T{job_id}', verticalalignment='bottom', fontsize=8)

    # 设置 X 轴刻度为每小时 (0-24 小时)
    # 固定范围：0-24 小时，间隔 1 小时
    ticks = np.arange(0, 25 * 60, 60)  # 0, 60, 120, ..., 1440 (24 小时)
        
    # 设置刻度标签为小时格式
    tick_labels = [f'{int(t//60)}h' for t in ticks]
        
    plt.xticks(ticks, tick_labels, rotation=45)
    plt.xlim(0, 24 * 60)  # 限制 X 轴范围为 0-24 小时
        
    plt.xlabel('Time (Hours)')
    plt.ylabel('Station')
    plt.title('Train Working Diagram (Time-Distance Graph)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 修正Y轴标签：0->A, 1->B, 2->C, 3->D, 4->E
    station_names = ['A', 'B', 'C', 'D', 'E']
    plt.yticks(range(len(station_names)), station_names)
    
    # 如果列车太多，图例可能会很乱，可以选择不显示
    if n_jobs <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Train graph saved to {save_path}")
    plt.close()

def plot_gantt_chart(schedule, n_machines, n_jobs, save_path='gantt_chart.png'):
    """
    绘制资源占用甘特图
    Y轴：机器（车站/区间）
    X轴：时间
    """
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_jobs))
    
    for (job_id, op_idx), (machine_id, start, end) in schedule.items():
        duration = end - start
        plt.barh(machine_id, duration, left=start, height=0.6, 
                 color=colors[job_id], edgecolor='black', alpha=0.8)
        
        # 在条形图中间标注车次
        if duration > 2:  # 只有足够宽才标注
            plt.text(start + duration/2, machine_id, f'T{job_id}', 
                     ha='center', va='center', color='white', fontsize=8)

    plt.xlabel('Time')
    plt.ylabel('Resource (Station / Section)')
    plt.title('Resource Occupation Gantt Chart')
    plt.yticks(range(n_machines), [f'M{i}' for i in range(n_machines)])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gantt chart saved to {save_path}")
    plt.close()

def save_schedule_to_csv(schedule, filename='results/train_schedule.csv'):
    """
    将调度结果保存为CSV（替代Excel以减少依赖）
    """
    data = []
    for (job_id, op_idx), (machine_id, start, end) in schedule.items():
        data.append({
            'Train ID': job_id,
            'Operation ID': op_idx,
            'Station/Section (Machine)': machine_id,
            'Start Time': start,
            'End Time': end,
            'Duration': end - start
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 按车次和开始时间排序
    df = df.sort_values(by=['Train ID', 'Start Time'])
    
    # 保存为CSV
    df.to_csv(filename, index=False, encoding='utf_8_sig')
    print(f"Schedule saved to {filename}")
