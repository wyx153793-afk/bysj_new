import torch
import numpy as np
import os
import glob
import re
import pandas as pd
from datetime import datetime
from JSP_Env import JSP_Env
from PPO_JSP import PPO_JSP
import matplotlib.pyplot as plt
from vis_utils import plot_train_graph, plot_gantt_chart, save_schedule_to_csv


def get_excel_data(excel_path='data/data4.xlsx'):
    """从 Excel 读取高铁调度基础数据"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到数据文件：{excel_path}。请确保存在 data 文件夹且文件名为 data*.xlsx")

    # 1. 读取全局参数 (Sheet1)
    df_global = pd.read_excel(excel_path, sheet_name='Sheet1')
    # 【新增代码】：强制清除 param_name 这一列中所有字符串前后的隐藏空格
    df_global['param_name'] = df_global['param_name'].astype(str).str.strip()
    df_global = df_global.set_index('param_name')
    # 获取不同类型列车的速度
    speeds = {
        'D': float(df_global.loc['speed_D', 'param_value']),
        'C': float(df_global.loc['speed_C', 'param_value']),
        'Z': float(df_global.loc['speed_Z', 'param_value']),
        'T': float(df_global.loc['speed_T', 'param_value']),
        'K': float(df_global.loc['speed_K', 'param_value']),
        'Normal': float(df_global.loc['speed_Normal', 'param_value']),
        'Freight': float(df_global.loc['speed_Freight', 'param_value'])
    }
    skylight_total_time = float(df_global.loc['skylight_total_time', 'param_value'])

    # 2. 读取区间信息 (Sheet2)
    df_sections = pd.read_excel(excel_path, sheet_name='Sheet2')
    total_dist = df_sections['distance'].sum()

    # 建立 机器ID -> 行驶时间 的映射
    machine_time_map = {t: {} for t in speeds.keys()}
    machine_skylight_time_map = {}

    for _, row in df_sections.iterrows():
        dist = row['distance']
        # 天窗时间 (按距离比例分配)
        skylight_time = int((dist / total_dist) * skylight_total_time)

        down_m = row['down_machine']
        up_m = row['up_machine']

        machine_skylight_time_map[down_m] = skylight_time
        machine_skylight_time_map[up_m] = skylight_time

        # 计算各类型列车的时间
        for t_type, speed in speeds.items():
            # 【修改】：因为速度本身就是 km/min，除完直接是分钟！千万不要乘 60 了
            # 保留 max(1, ...) 是为了防止距离太短(如小于速度值)时 int() 取整变成 0
            time_val = max(1, int(dist / speed))
            machine_time_map[t_type][down_m] = time_val
            machine_time_map[t_type][up_m] = time_val

    # 3. 读取列车任务信息 (Sheet3)
    try:
        df_trains = pd.read_excel(excel_path, sheet_name='Sheet3')
    except ValueError:
        try:
            df_trains = pd.read_excel(excel_path, sheet_name='sheet3')
        except ValueError:
            df_trains = pd.read_excel(excel_path, sheet_name='Trains')

    n_jobs = len(df_trains)

    def parse_sequence(seq_val):
        seq_str = str(seq_val).replace('，', ',').replace(' ', '')
        if '-' in seq_str or ':' in seq_str:
            raise ValueError(f"数据读取错误：识别到类似日期的格式 '{seq_str}'。")
        return [int(m) for m in seq_str.split(',')]

    all_seqs = df_trains['machine_seq'].apply(parse_sequence)
    max_ops = all_seqs.apply(len).max()

    all_machines = set()
    for seq in all_seqs:
        all_machines.update(seq)
    n_machines = max(all_machines) + 1

    # 4. 初始化输出矩阵与天窗列表
    processing_time = np.zeros((n_jobs, max_ops), dtype=int)
    op_machine_assign = -np.ones((n_jobs, max_ops), dtype=int)
    maintenance_job_ids = []

    def get_train_type(train_name):
        train_name = str(train_name).strip()
        if train_name == 'TC0': return 'Skylight'
        if re.match(r'^D\d{3}$', train_name): return 'D'
        if re.match(r'^C\d{3}$', train_name): return 'C'
        if re.match(r'^Z\d{3,4}$', train_name): return 'Z'
        if re.match(r'^T\d{3,4}$', train_name): return 'T'
        if re.match(r'^K\d{4,5}$', train_name): return 'K'
        if re.match(r'^\d{1,4}$', train_name): return 'Normal'
        if re.match(r'^\d{5}$', train_name) or re.match(r'^X\d{4}$', train_name): return 'Freight'
        return 'Normal'  # 默认降级为普通列车

    # 5. 填充矩阵数据并增加安全校验
    for idx, row in df_trains.iterrows():
        job_id = int(row['job_id'])
        train_name = row['train_name']

        # 判断类型
        t_type = get_train_type(train_name)
        is_skylight = (t_type == 'Skylight')

        if is_skylight:
            maintenance_job_ids.append(job_id)

        seq = all_seqs.iloc[idx]

        for op_idx, m_id in enumerate(seq):
            op_machine_assign[job_id, op_idx] = m_id
            if is_skylight:
                processing_time[job_id, op_idx] = machine_skylight_time_map[m_id]
            else:
                processing_time[job_id, op_idx] = machine_time_map[t_type][m_id]

            # ================= 新增：6. 读取停站时间 (Sheet4) =================
            # 根据机器数推算车站数
        n_stations = (n_machines // 2) + 1
        min_stop_times = np.zeros((n_jobs, n_stations), dtype=int)

        # 建立 train_name 到 job_id 的映射，方便后面填入对应位置
        train_to_job = {}
        for idx, row in df_trains.iterrows():
            t_name = str(row['train_name']).strip()
            train_to_job[t_name] = int(row['job_id'])

        try:
            df_stops = pd.read_excel(excel_path, sheet_name='Sheet4')
            # 获取所有以 station 开头的列名（忽略大小写）
            station_cols = [c for c in df_stops.columns if str(c).lower().startswith('station')]

            for idx, row in df_stops.iterrows():
                t_name = str(row['train_name']).strip()
                if t_name in train_to_job:
                    j_id = train_to_job[t_name]
                    # 按顺序遍历车站列，填入矩阵
                    for i, col in enumerate(station_cols):
                        if i < n_stations:  # 防止越界
                            val = row[col]
                            # 遇到空值(NaN)填0，否则转为整数
                            min_stop_times[j_id, i] = 0 if pd.isna(val) else int(val)
        except Exception as e:
            print(f"⚠️ 警告：读取 Sheet4 停站时间失败 ({e})，将默认所有停站时间为 0。")
        # ==================================================================

        # 【修改】：返回的最后加上 min_stop_times
        return processing_time, op_machine_assign, n_jobs, n_machines, max_ops, maintenance_job_ids, min_stop_times

def find_best_historical_model(directories):
    """
    扫描指定的文件夹列表，解析文件名，返回 makespan 最小的模型路径
    """
    best_path = None
    global_best_makespan = float('inf')
    # 正则匹配文件名中的 makespan 数值
    pattern = re.compile(r'jsp_best_mk_([0-9.]+)_')

    for directory in directories:
        if not os.path.exists(directory):
            continue
        for filepath in glob.glob(os.path.join(directory, '*.pth')):
            filename = os.path.basename(filepath)
            match = pattern.search(filename)
            if match:
                mk_val = float(match.group(1))
                # 寻找 makespan 最小的（时间越短越优）
                if mk_val < global_best_makespan:
                    global_best_makespan = mk_val
                    best_path = filepath

    return best_path, global_best_makespan

def train():
    # ==========================================
    # 模式切换区
    # ==========================================
    USE_FIXED_EPISODES = False  # True: 固定轮数; False: 无限轮数到最优即停
    n_episodes = 500           # 固定模式下的总轮数
    patience = 50              # 无限模式下：连续多少轮没破记录则早停

    # 自动加载历史最优模型开关
    LOAD_PREVIOUS_BEST = False
    # ==========================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/train_{timestamp}'

    # 为两种训练模式分配不同的最优模型文件夹
    dir_fixed = 'results/best_models_fixed'
    dir_infinite = 'results/best_models_infinite'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dir_fixed, exist_ok=True)
    os.makedirs(dir_infinite, exist_ok=True)

    current_best_dir = dir_fixed if USE_FIXED_EPISODES else dir_infinite

    print(f"Training Mode: {'Fixed Episodes' if USE_FIXED_EPISODES else 'Train Until Optimal'}")
    print(f"Current best models will be saved to: {current_best_dir}")

    # 调用修改后的 get_excel_data 函数
    try:
        # 注意：这里等号左边现在是 7 个变量了。
        proc_times, machine_assign, n_jobs, n_machines, max_ops, maintenance_job_ids, min_stop_times = get_excel_data(
            'data/data4.xlsx')

        print(f"成功加载 Excel 数据！总任务数: {n_jobs}, 机器数: {n_machines}")
        print(f"识别到天窗列车 ID: {maintenance_job_ids}")
    except Exception as e:
        # 💡 调试小贴士：如果以后遇到报错不知道在哪，可以把下面这行改成 raise e，这样就会打印详细报错行号
        raise e
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo = PPO_JSP(n_jobs, n_machines, device=device)

    # ==========================================
    # 调用二者中更优的那个模型
    # ==========================================
    if LOAD_PREVIOUS_BEST:
        best_hist_path, best_hist_mk = find_best_historical_model([dir_fixed, dir_infinite])
        if best_hist_path:
            print(f"🚀 Found historical best model across all modes!")
            print(f"   Path: {best_hist_path}")
            print(f"   Best Makespan: {best_hist_mk}")
            try:
                ppo.load(best_hist_path)
                print("   Successfully loaded! Continuing training from this elite state...\n")
            except Exception as e:
                print(f"   Failed to load existing model: {e}\n")
        else:
            print("No historical models found. Starting training from scratch.\n")

    session_best_makespan = float('inf')
    session_best_model_path = None

    makespan_history = []
    last_env = None
    episodes_without_improvement = 0
    episode = 0

    while True:
        if USE_FIXED_EPISODES and episode >= n_episodes:
            print(f"Reached fixed maximum episodes ({n_episodes}). Stopping.")
            break

        # 【核心修改】将动态读取的 min_stop_times 传给环境
        env = JSP_Env(n_jobs, n_machines, proc_times, machine_assign,
                      maintenance_job_ids=maintenance_job_ids,
                      min_stop_times=min_stop_times,
                      device=device)
        state = env.reset()
        done = False

        while not done:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, info = env.step(action)
            ppo.store_transition(state, action, reward, next_state, done, log_prob)
            state = next_state

        last_env = env
        makespan = env.get_makespan()
        makespan_history.append(makespan)

        # 发现本次训练的新最优解
        if makespan < session_best_makespan:
            session_best_makespan = makespan
            episodes_without_improvement = 0

            new_save_path = os.path.join(current_best_dir, f'jsp_best_mk_{makespan}_{timestamp}.pth')

            if session_best_model_path and os.path.exists(session_best_model_path):
                try:
                    os.remove(session_best_model_path)
                except OSError as e:
                    print(f"Warning: Failed to delete previous best model file: {e}")

            ppo.save(new_save_path)
            session_best_model_path = new_save_path
            print(f"🟢 Episode {episode}: New session best makespan {makespan} found! Saved to {new_save_path}")
        else:
            episodes_without_improvement += 1

        if episode % 10 == 0 and episode > 0:
            ppo.update()

        if episode % 50 == 0:
            avg_makespan = np.mean(makespan_history[-50:]) if len(makespan_history) >= 50 else np.mean(makespan_history)
            print(f"Episode {episode}, Current: {makespan}, Avg(50): {avg_makespan:.2f}, No Improve: {episodes_without_improvement}")

        if not USE_FIXED_EPISODES and episodes_without_improvement >= patience:
            print(f"🔴 No improvement for {patience} episodes. Early stopping at optimal conditions.")
            break

        episode += 1

    current_model_path = os.path.join(results_dir, 'policy_job_final.pth')
    ppo.save(current_model_path)
    np.save(os.path.join(results_dir, 'training_history.npy'), makespan_history)

    if last_env:
        plot_train_graph(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'train_graph.png'))
        plot_gantt_chart(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'gantt_chart.png'))
        save_schedule_to_csv(last_env.schedule, os.path.join(results_dir, 'train_schedule.csv'))

    plt.figure()
    plt.plot(makespan_history)
    plt.xlabel('Episode')
    plt.ylabel('Makespan (Excluding Skylight)')
    plt.title(f'JSP Training Curve ({timestamp})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(results_dir, 'training_curve.png'))
    plt.close()

if __name__ == '__main__':
    train()