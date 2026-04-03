"""
JSP Environment for Train Timetabling Problem
"""
import torch
import numpy as np

class JSP_Env:
    def __init__(self, n_jobs, n_machines, processing_time_matrix,
                 job_op_sequences, maintenance_job_ids=None, train_priorities=None, station_capacities=None,
                 planned_departures=None, min_stop_times=None, device='cuda'):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device

        self.processing_time = processing_time_matrix
        self.op_machine_assign = job_op_sequences

        self.max_ops = processing_time_matrix.shape[1]

        # --- 动态推算区间与车站数量 ---
        self.n_sections = self.n_machines // 2
        self.n_stations = self.n_sections + 1

        # 【修改点】：追踪与天窗基础参数，根据铁路行车组织规则，V型天窗前后列车安全间隔设置为 15 分钟
        self.min_headway_maintenance = 15

        # 而是使用外部传入的列表。如果没有传，则默认为空列表 []
        self.maintenance_job_ids = maintenance_job_ids if maintenance_job_ids is not None else []

        # 车站安全间隔时间约束 (单位: min)
        self.interval_DD = 240 / 60.0  # 到到 4.0 min
        self.interval_DT = 180 / 60.0  # 到通 3.0 min
        self.interval_FF = 240 / 60.0  # 发发 4.0 min
        self.interval_FT = 300 / 60.0  # 发通 5.0 min
        self.interval_TT = 180 / 60.0  # 通通 3.0 min
        self.interval_TF = 90 / 60.0   # 通发 1.5 min
        self.interval_TD = 240 / 60.0  # 通到 4.0 min

        # 【新增】：建立状态组合与安全间隔的映射字典，用于动态查询
        self.intervals = {
            'FF': self.interval_FF, 'FT': self.interval_FT, 'TF': self.interval_TF, 'TT': self.interval_TT,
            'DD': self.interval_DD, 'DT': self.interval_DT, 'TD': self.interval_TD
        }

        # --- 动态生成机器ID对应的物理起止车站映射 ---
        self.machine_to_stations = {}
        for m in range(self.n_sections):
            # 下行机器映射 (如 0 对应车站 0->1)
            self.machine_to_stations[m] = (m, m + 1)
            # 上行机器映射 (基于对称性，如 44台机器时，43 对应车站 1->0)
            opp_m = self.n_machines - 1 - m
            self.machine_to_stations[opp_m] = (m + 1, m)

        # 1. 列车等级权重 Wi
        self.W = train_priorities if train_priorities is not None else np.ones(n_jobs)

        # 2. 动态生成车站到发线容量 (默认为每个车站2条到发线)
        self.station_capacities = station_capacities if station_capacities is not None else {i: 2 for i in range(self.n_stations)}

        # 3. 计划始发时间
        self.planned_departures = planned_departures

        # 4. 动态生成最短停站时间矩阵
        self.min_stop_times = min_stop_times if min_stop_times is not None else np.zeros((n_jobs, self.n_stations))

        self.reset()

    def reset(self):
        self.current_op_idx = np.zeros(self.n_jobs, dtype=int)
        self.machine_available_time = np.zeros(self.n_machines)
        self.machine_last_job = -np.ones(self.n_machines, dtype=int)
        self.machine_last_leave_time = np.zeros(self.n_machines)
        self.machine_last_enter_time = np.zeros(self.n_machines)
        self.job_ready_time = np.zeros(self.n_jobs)
        self.completed_ops = np.zeros((self.n_jobs, self.max_ops), dtype=bool)
        self.completed_jobs = np.zeros(self.n_jobs, dtype=bool)

        # ======== 【新增防御 1：初始化时剔除无动作的无效列车】 ========
        for j in range(self.n_jobs):
            if self.max_ops == 0 or self.op_machine_assign[j, 0] == -1:
                self.completed_jobs[j] = True
        # ==========================================================

        self.schedule = {}
        self.actual_departures = np.zeros(self.n_jobs)

        # 动态适配字典长度
        self.station_records = {i: [] for i in range(self.n_stations)}
        self.section_records = {i: [] for i in range(self.n_machines)}

        return self.get_state()

    def check_capacity_violation(self, st, new_start, new_end):
        events = []
        for s, e in self.station_records[st]:
            if e > s:
                events.append((s, 1))
                events.append((e, -1))
        if new_end > new_start:
            events.append((new_start, 1))
            events.append((new_end, -1))

        events.sort(key=lambda x: (x[0], x[1]))
        max_concurrent = 0
        current = 0
        for t, change in events:
            current += change
            if current > max_concurrent:
                max_concurrent = current

        return max_concurrent > self.station_capacities.get(st, 2)

    def get_state(self):
        ready_ops_features = []
        for job_id in range(self.n_jobs):
            if self.completed_jobs[job_id]:
                ready_ops_features.append([0, 0, 0, 0])
            else:
                op_idx = self.current_op_idx[job_id]
                machine_id = self.op_machine_assign[job_id, op_idx]
                proc_time = self.processing_time[job_id, op_idx]
                ready_ops_features.append([
                    job_id / self.n_jobs,
                    op_idx / self.max_ops,
                    machine_id / self.n_machines,
                    proc_time / 100.0
                ])

        machine_features = []
        for m in range(self.n_machines):
            machine_features.append([
                self.machine_available_time[m] / 1000.0,
                0
            ])

        return {
            'ready_ops': torch.FloatTensor(ready_ops_features).to(self.device),
            'machines': torch.FloatTensor(machine_features).to(self.device),
            'mask': self.get_action_mask()
        }

    def get_action_mask(self):
        mask = np.zeros(self.n_jobs, dtype=bool)
        for job_id in range(self.n_jobs):
            if not self.completed_jobs[job_id]:
                mask[job_id] = True
        return torch.BoolTensor(mask).to(self.device)

    def get_makespan(self):
        regular_jobs_ready_time = []
        for j in range(self.n_jobs):
            if j not in self.maintenance_job_ids:
                regular_jobs_ready_time.append(self.job_ready_time[j])
        if not regular_jobs_ready_time:
            return 0
        return max(regular_jobs_ready_time)

    def step(self, job_id):
        job_id = int(job_id)  # 确保传入的是整数索引

        # ======== 【新增防御 2：严禁网络选择已完成的列车】 ========
        if self.completed_jobs[job_id]:
            # 如果网络瞎选，给一个巨大的惩罚，并强制结束或返回原状态
            done = np.all(self.completed_jobs)
            return self.get_state(), -10.0, done, {'makespan': self.get_makespan(), 'scheduled': None}
        # ==========================================================

        op_idx = self.current_op_idx[job_id]
        machine_id = self.op_machine_assign[job_id, op_idx]

        # ======== 【新增防御 3：最后一道防线，兜底拦截 -1】 ========
        if machine_id == -1:
            self.completed_jobs[job_id] = True
            done = np.all(self.completed_jobs)
            return self.get_state(), -10.0, done, {'makespan': self.get_makespan(), 'scheduled': None}
        # ==========================================================

        proc_time = self.processing_time[job_id, op_idx]

        # 动态计算对向机器ID
        opp_machine_id = self.n_machines - 1 - machine_id
        is_curr_maintenance = (job_id in self.maintenance_job_ids)

        ready_t = self.job_ready_time[job_id]
        start_st, end_st = self.machine_to_stations[machine_id]

        min_stop = self.min_stop_times[job_id, start_st] if op_idx > 0 else 0
        start_time = ready_t + min_stop

        # 初始启发式快速推进（保留以加速寻优）
        last_j = self.machine_last_job[machine_id]
        if last_j != -1:
            last_enter = self.machine_last_enter_time[machine_id]
            last_leave = self.machine_last_leave_time[machine_id]

            if last_j in self.maintenance_job_ids or is_curr_maintenance:
                start_time = max(start_time, last_leave + self.min_headway_maintenance)
            else:
                dir_curr = job_id % 2
                dir_last = last_j % 2
                if dir_curr == dir_last:
                    hw = max(self.interval_FF, self.interval_FT, self.interval_TF, self.interval_TT)
                else:
                    hw = max(self.interval_DD, self.interval_DT, self.interval_TD)
                start_time = max(start_time, last_enter + hw)

        if is_curr_maintenance:
            last_j_opp = self.machine_last_job[opp_machine_id]
            if last_j_opp != -1:
                start_time = max(start_time, self.machine_last_leave_time[opp_machine_id] + self.min_headway_maintenance)

        conflict = True
        while conflict:
            conflict = False

            # 【核心修改】：精准识别当前时刻下的发、到、通状态
            # 出发站(s)：若是首站、或有最短停站时间要求、或实际发生等待（start_time > ready_t），均为"发(F)"，否则为"通(T)"
            curr_event_start = 'F' if (op_idx == 0 or min_stop > 0 or start_time > ready_t) else 'T'
            # 到达站(s+1)：若是终到站、或下一站有最短停站时间要求，均为"到(D)"，否则为"通(T)"
            curr_event_end = 'D' if (op_idx == self.max_ops - 1 or self.min_stop_times[job_id, end_st] > 0) else 'T'

            end_time = start_time + proc_time

            # 遍历区间内所有已安排列车，进行严格次序与安全间隔约束判定
            for s_prev, e_prev, j_prev, prev_event_start, prev_event_end in self.section_records[machine_id]:
                is_j_maint = (j_prev in self.maintenance_job_ids)
                if is_curr_maintenance or is_j_maint:
                    hw = self.min_headway_maintenance
                    if not (end_time <= s_prev or start_time >= e_prev + hw):
                        start_time = max(start_time, e_prev + hw)
                        conflict = True
                        break
                else:
                    # 【核心修改】：代入文献中的列车运行图定序优化数学不等式约束
                    if start_time >= s_prev:
                        # 当前列车排在 j_prev 之后

                        # 1. 出发安全间隔约束：d_j^s - d_i^s > HW_start
                        hw_start = self.intervals[prev_event_start + curr_event_start]
                        if start_time < s_prev + hw_start:
                            start_time = s_prev + hw_start
                            conflict = True
                            break

                        # 2. 到达安全间隔约束：a_j^{s+1} - a_i^{s+1} > HW_end
                        hw_end = self.intervals[prev_event_end + curr_event_end]
                        if end_time < e_prev + hw_end:
                            # 为保证到达间隔满足要求，逆推推迟发车时间
                            start_time = e_prev + hw_end - proc_time
                            conflict = True
                            break
                    else:
                        # 当前列车试图排在 j_prev 之前

                        # 1. 出发安全间隔约束：d_i^s - d_j^s > HW_start (此时j_prev成为后车)
                        hw_start = self.intervals[curr_event_start + prev_event_start]
                        if s_prev < start_time + hw_start:
                            # 空间不足以插入，被迫延后到 j_prev 之后发车
                            start_time = s_prev + self.intervals[prev_event_start + curr_event_start]
                            conflict = True
                            break

                        # 2. 到达安全间隔约束：a_i^{s+1} - a_j^{s+1} > HW_end
                        hw_end = self.intervals[curr_event_end + prev_event_end]
                        if e_prev < end_time + hw_end:
                            # 区间不可越行约束，被迫延后到 j_prev 之后发车
                            start_time = s_prev + self.intervals[prev_event_start + curr_event_start]
                            conflict = True
                            break

            if conflict: continue

            # 对向线路天窗冲突判断
            for s_prev, e_prev, j_prev, _, _ in self.section_records[opp_machine_id]:
                is_j_maint = (j_prev in self.maintenance_job_ids)
                if is_curr_maintenance or is_j_maint:
                    hw = self.min_headway_maintenance
                    if not (end_time <= s_prev or start_time >= e_prev + hw):
                        start_time = max(start_time, e_prev + hw)
                        conflict = True
                        break

        # 确定下最终发车时间后，确立最终端点事件状态
        final_event_start = 'F' if (op_idx == 0 or min_stop > 0 or start_time > ready_t) else 'T'
        final_event_end = 'D' if (op_idx == self.max_ops - 1 or self.min_stop_times[job_id, end_st] > 0) else 'T'
        end_time = start_time + proc_time

        waiting_time = max(0, start_time - ready_t - min_stop)
        step_penalty = 0.0

        if start_time > ready_t:
            if self.check_capacity_violation(start_st, ready_t, start_time):
                step_penalty -= 500.0
            self.station_records[start_st].append((ready_t, start_time))

        if op_idx == 0 and not is_curr_maintenance:
            self.actual_departures[job_id] = start_time

        # 【核心修改】：section_records 新增记录当前作业的发、到、通状态元组 (s, e, j, event_s, event_e)
        self.section_records[machine_id].append((start_time, end_time, job_id, final_event_start, final_event_end))

        if is_curr_maintenance:
            self.section_records[opp_machine_id].append((start_time, end_time, job_id, final_event_start, final_event_end))

        self.schedule[(job_id, op_idx)] = (machine_id, start_time, end_time)

        self.machine_last_job[machine_id] = job_id
        self.machine_last_leave_time[machine_id] = end_time
        self.machine_last_enter_time[machine_id] = start_time
        self.machine_available_time[machine_id] = end_time

        if is_curr_maintenance:
            self.machine_last_job[opp_machine_id] = job_id
            self.machine_last_leave_time[opp_machine_id] = end_time
            self.machine_last_enter_time[opp_machine_id] = start_time
            self.machine_available_time[opp_machine_id] = end_time

        self.job_ready_time[job_id] = end_time
        self.completed_ops[job_id, op_idx] = True

        self.current_op_idx[job_id] += 1

        if self.current_op_idx[job_id] >= self.max_ops or \
                self.processing_time[job_id, self.current_op_idx[job_id]] == 0 or \
                self.op_machine_assign[job_id, self.current_op_idx[job_id]] == -1:
            self.completed_jobs[job_id] = True

        done = np.all(self.completed_jobs)

        if not is_curr_maintenance:
            z1_component = self.W[job_id] * (proc_time + min_stop + waiting_time)
            z3_component = waiting_time
            step_reward = - (z1_component + z3_component) / 100.0
        else:
            step_reward = 0

        reward = step_reward + step_penalty

        if done:
            z2_penalty = 0.0
            reg_deps = [self.actual_departures[j] for j in range(self.n_jobs) if j not in self.maintenance_job_ids]
            if len(reg_deps) > 1:
                reg_deps = np.sort(reg_deps)
                z2_penalty = np.sum(np.abs(np.diff(reg_deps)))

            reward -= (z2_penalty / 100.0)

        return self.get_state(), reward, done, {
            'makespan': self.get_makespan(),
            'scheduled': (job_id, op_idx, machine_id, start_time, end_time) if not done else None
        }