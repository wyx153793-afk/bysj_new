import torch
import numpy as np
import os
import glob
import re
import pandas as pd
import copy
from datetime import datetime
from JSP_Env import JSP_Env
from PPO_JSP import PPO_JSP
import matplotlib.pyplot as plt
from vis_utils import plot_train_graph, plot_gantt_chart, save_schedule_to_csv


def get_excel_data(excel_path='data/data5.xlsx'):
    """从 Excel 读取高铁调度基础数据"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到数据文件：{excel_path}。请确保存在 data 文件夹且文件名为 data*.xlsx")

    # 1. 读取全局参数
    df_global = pd.read_excel(excel_path, sheet_name='Sheet1')
    df_global['param_name'] = df_global['param_name'].astype(str).str.strip()
    df_global = df_global.set_index('param_name')
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

    # 2. 读取区间信息
    df_sections = pd.read_excel(excel_path, sheet_name='Sheet2')
    total_dist = df_sections['distance'].sum()

    machine_time_map = {t: {} for t in speeds.keys()}
    machine_skylight_time_map = {}

    for _, row in df_sections.iterrows():
        dist = row['distance']
        # 天窗时间 (按距离比例分配)
        if total_dist > 0:
            skylight_time = int((dist / total_dist) * skylight_total_time)
        else:
            skylight_time = 0

        down_m = row['down_machine']
        up_m = row['up_machine']

        machine_skylight_time_map[down_m] = skylight_time
        machine_skylight_time_map[up_m] = skylight_time

        for t_type, speed in speeds.items():

            if dist <= 0.0001:
                time_val = 0
            else:
                time_val = max(1, int(dist / speed))
            machine_time_map[t_type][down_m] = time_val
            machine_time_map[t_type][up_m] = time_val

    # 3. 读取列车任务信息
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

    # 4. 初始化输出矩阵
    processing_time = np.zeros((n_jobs, max_ops), dtype=int)
    op_machine_assign = -np.ones((n_jobs, max_ops), dtype=int)
    maintenance_job_ids = []

    # 车次字典用于绘图标识
    train_names_dict = {}
    train_priorities = np.ones(n_jobs, dtype=float)

    weight_map = {
        'D': 2.0, 'C': 2.0, 'Z': 1.5, 'T': 1.2,
        'K': 1.0, 'Normal': 0.8, 'Freight': 0.5, 'Skylight': 1.0
    }

    def get_train_type(train_name):
        train_name = str(train_name).strip().upper()
        if train_name.startswith('TC'): return 'Skylight'
        if re.match(r'^D\d{3}$', train_name): return 'D'
        if re.match(r'^C\d{3}$', train_name): return 'C'
        if re.match(r'^Z\d{3,4}$', train_name): return 'Z'
        if re.match(r'^T\d{3,4}$', train_name): return 'T'
        if re.match(r'^K\d{4,5}$', train_name): return 'K'
        if re.match(r'^\d{1,4}$', train_name): return 'Normal'
        if re.match(r'^\d{5}$', train_name) or re.match(r'^X\d{4}$', train_name): return 'Freight'
        return 'Normal'

    # 5. 填充数据
    for idx, row in df_trains.iterrows():
        job_id = int(row['job_id'])
        t_name_raw = str(row['train_name']).strip()

        train_names_dict[job_id] = t_name_raw

        t_type = get_train_type(t_name_raw)
        is_skylight = (t_type == 'Skylight')

        train_priorities[job_id] = weight_map[t_type]

        if is_skylight:
            maintenance_job_ids.append(job_id)

        seq = all_seqs.iloc[idx]

        for op_idx, m_id in enumerate(seq):
            op_machine_assign[job_id, op_idx] = m_id
            if is_skylight:
                processing_time[job_id, op_idx] = machine_skylight_time_map[m_id]
            else:
                processing_time[job_id, op_idx] = machine_time_map[t_type][m_id]

    # 6. 读取停站时间
    n_stations = (n_machines // 2) + 1
    min_stop_times = np.zeros((n_jobs, n_stations), dtype=int)

    train_to_job = {}
    for idx, row in df_trains.iterrows():
        t_name = str(row['train_name']).strip()
        train_to_job[t_name] = int(row['job_id'])

    try:
        df_stops = pd.read_excel(excel_path, sheet_name='Sheet4')
        station_cols = [c for c in df_stops.columns if str(c).lower().startswith('station')]

        for idx, row in df_stops.iterrows():
            t_name = str(row['train_name']).strip()
            if t_name in train_to_job:
                j_id = train_to_job[t_name]
                for i, col in enumerate(station_cols):
                    if i < n_stations:
                        val = row[col]
                        min_stop_times[j_id, i] = 0 if pd.isna(val) else int(val)
    except Exception as e:
        print(f"⚠️ 警告：读取 Sheet4 停站时间失败 ({e})，将默认所有停站时间为 0。")

    return processing_time, op_machine_assign, n_jobs, n_machines, max_ops, maintenance_job_ids, min_stop_times, train_priorities, train_names_dict


def find_best_historical_model(directories):
    best_path = None
    global_best_makespan = float('inf')
    pattern = re.compile(r'jsp_best_mk_([0-9.]+)_')

    for directory in directories:
        if not os.path.exists(directory):
            continue
        for filepath in glob.glob(os.path.join(directory, '*.pth')):
            filename = os.path.basename(filepath)
            match = pattern.search(filename)
            if match:
                mk_val = float(match.group(1))
                if mk_val < global_best_makespan:
                    global_best_makespan = mk_val
                    best_path = filepath

    return best_path, global_best_makespan


def train():
    USE_FIXED_EPISODES = True
    n_episodes = 300
    LOAD_PREVIOUS_BEST = False

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    window_size = 100
    min_improvement = 30
    patience = 200
    best_sliding_avg = float('inf')
    no_improve_count = 0
    results_dir = f'results/train_{timestamp}'
    dir_fixed = 'results/best_models_fixed'
    dir_infinite = 'results/best_models_infinite'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dir_fixed, exist_ok=True)
    os.makedirs(dir_infinite, exist_ok=True)

    current_best_dir = dir_fixed if USE_FIXED_EPISODES else dir_infinite

    print(f"Training Mode: {'Fixed Episodes' if USE_FIXED_EPISODES else 'Train Until Optimal'}")

    try:
        proc_times, machine_assign, n_jobs, n_machines, max_ops, maintenance_job_ids, min_stop_times, train_priorities, train_names_dict = get_excel_data(
            'data/data5.xlsx')

        print(f"成功加载 Excel 数据！总任务数: {n_jobs}, 机器数: {n_machines}")
        print(f"识别到天窗列车 ID: {maintenance_job_ids}")
    except Exception as e:
        raise e
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo = PPO_JSP(n_jobs, n_machines, lr=1e-4, device=device)

    if LOAD_PREVIOUS_BEST:
        best_hist_path, best_hist_mk = find_best_historical_model([dir_fixed, dir_infinite])
        if best_hist_path:
            print(f"🚀 Found historical best model across all modes!")
            try:
                ppo.load(best_hist_path)
            except Exception as e:
                pass

    session_best_makespan = float('inf')
    session_best_model_path = None
    makespan_history = []
    last_env = None
    best_env = None
    episodes_without_improvement = 0
    episode = 0

    while True:
        if USE_FIXED_EPISODES and episode >= n_episodes:
            break

        env = JSP_Env(n_jobs, n_machines, proc_times, machine_assign,
                      maintenance_job_ids=maintenance_job_ids,
                      train_priorities=train_priorities,
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
        # 每50轮评估一次滑动窗口均值
        if episode > 0 and episode % 50 == 0 and len(makespan_history) >= window_size:
            recent_avg = np.mean(makespan_history[-window_size:])

            if recent_avg < best_sliding_avg - min_improvement:
                best_sliding_avg = recent_avg
                no_improve_count = 0
                print(f"🟢 Episode {episode}: 滑动均值改进至 {recent_avg:.1f}")
            else:
                no_improve_count += 50
                print(f"🟡 Episode {episode}: 滑动均值 {recent_avg:.1f}, 无改进: {no_improve_count}/{patience}")

            # 智能早停
            if no_improve_count >= patience:
                print(f"🔴 早停: {patience}轮无显著改进")
                print(f"   最优滑动均值: {best_sliding_avg:.1f}")
                break


        if makespan < session_best_makespan:
            session_best_makespan = makespan
            episodes_without_improvement = 0
            best_env = copy.deepcopy(env)

            new_save_path = os.path.join(current_best_dir, f'jsp_best_mk_{makespan}_{timestamp}.pth')
            if session_best_model_path and os.path.exists(session_best_model_path):
                try:
                    os.remove(session_best_model_path)
                except OSError:
                    pass

            ppo.save(new_save_path)
            session_best_model_path = new_save_path
            print(f"🟢 Episode {episode}: New session best makespan {makespan} found!")
        else:
            episodes_without_improvement += 1

        # 积累足够样本就更新，更频繁
        if len(ppo.buffer) >= 32:  # 或每轮都更新：if len(ppo.buffer) > 0:
            ppo.update()

        if episode % 50 == 0:
            avg_makespan = np.mean(makespan_history[-50:]) if len(makespan_history) >= 50 else np.mean(makespan_history)
            print(
                f"Episode {episode}, Current: {makespan}, Avg(50): {avg_makespan:.2f}, No Improve: {episodes_without_improvement}")


        episode += 1

    current_model_path = os.path.join(results_dir, 'policy_job_final.pth')
    ppo.save(current_model_path)
    np.save(os.path.join(results_dir, 'training_history.npy'), makespan_history)

    if best_env:
        print(f"正在生成最优轮次 (Makespan={session_best_makespan}) 的图表和CSV...")
        plot_train_graph(best_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'train_graph_best.png'),
                         train_names_dict, maintenance_job_ids)
        plot_gantt_chart(best_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'gantt_chart_best.png'),
                         train_names_dict)
        save_schedule_to_csv(best_env.schedule, os.path.join(results_dir, 'train_schedule_best.csv'), train_names_dict)
    elif last_env:
        plot_train_graph(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'train_graph.png'),
                         train_names_dict, maintenance_job_ids)
        plot_gantt_chart(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'gantt_chart.png'),
                         train_names_dict)
        save_schedule_to_csv(last_env.schedule, os.path.join(results_dir, 'train_schedule.csv'), train_names_dict)

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