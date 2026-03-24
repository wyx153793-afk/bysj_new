"""
JSP Environment for Train Timetabling Problem
"""
import torch
import numpy as np

class JSP_Env:
    def __init__(self, n_jobs, n_machines, processing_time_matrix,
                 job_op_sequences, device='cuda'):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device

        self.processing_time = processing_time_matrix
        self.op_machine_assign = job_op_sequences

        self.max_ops = processing_time_matrix.shape[1]

        # 追踪与天窗基础参数
        self.min_headway_maintenance = 120 # 天窗列车安全间隔 120min
        self.maintenance_job_ids = [8, 9]  # T9, T10 为天窗列车

        # 间隔时间约束
        self.tau_bu = 3
        self.tau_hui = 2
        self.tau_lian = 4
        self.tau_daofa = 2
        self.tau_fadao = 5
        self.tau_tong = 4

        # 机器ID对应的物理起止车站映射 (0:A, 1:B, 2:C, 3:D, 4:E)
        self.machine_to_stations = {
            0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 4),
            4: (4, 3), 5: (3, 2), 6: (2, 1), 7: (1, 0)
        }

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
        self.schedule = {}

        self.station_records = {i: [] for i in range(5)}
        self.section_records = {i: [] for i in range(8)}

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
        return max_concurrent > 2

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

    def _compute_reward(self):
        if np.all(self.completed_jobs):
            return -self.get_makespan() / 100.0
        return 0.0

    def step(self, job_id):
        op_idx = self.current_op_idx[job_id]
        machine_id = self.op_machine_assign[job_id, op_idx]
        proc_time = self.processing_time[job_id, op_idx]

        opp_machine_id = 7 - machine_id
        is_curr_maintenance = (job_id in self.maintenance_job_ids)

        start_time = self.job_ready_time[job_id]

        last_j = self.machine_last_job[machine_id]
        if last_j != -1:
            last_enter = self.machine_last_enter_time[machine_id]
            last_leave = self.machine_last_leave_time[machine_id]
            if last_j in self.maintenance_job_ids:
                start_time = max(start_time, last_leave + self.min_headway_maintenance)
            else:
                dir_curr = job_id % 2
                dir_last = last_j % 2
                hw = max(self.tau_lian, self.tau_fadao) if dir_curr == dir_last else max(self.tau_bu, self.tau_hui, self.tau_tong)
                start_time = max(start_time, last_enter + hw)

        if is_curr_maintenance:
            last_j_opp = self.machine_last_job[opp_machine_id]
            if last_j_opp != -1:
                start_time = max(start_time, self.machine_last_leave_time[opp_machine_id])

        conflict = True
        while conflict:
            conflict = False
            end_time = start_time + proc_time

            for s, e, j in self.section_records[machine_id]:
                is_j_maint = (j in self.maintenance_job_ids)
                if is_curr_maintenance or is_j_maint:
                    hw = self.min_headway_maintenance if is_j_maint else 0
                    if not (end_time <= s or start_time >= e + hw):
                        start_time = max(start_time, e + hw)
                        conflict = True
                        break
                else:
                    dir_curr = job_id % 2
                    dir_j = j % 2
                    hw = max(self.tau_lian, self.tau_fadao) if dir_curr == dir_j else max(self.tau_bu, self.tau_hui, self.tau_tong)

                    if s <= start_time and e > end_time:
                        start_time = max(start_time, e + hw - proc_time)
                        conflict = True
                        break
                    elif start_time < s and end_time > e:
                        start_time = max(start_time, e + hw)
                        conflict = True
                        break
            if conflict: continue

            for s, e, j in self.section_records[opp_machine_id]:
                is_j_maint = (j in self.maintenance_job_ids)
                if is_curr_maintenance or is_j_maint:
                    hw = self.min_headway_maintenance if is_j_maint else 0
                    if not (end_time <= s or start_time >= e + hw):
                        start_time = max(start_time, e + hw)
                        conflict = True
                        break

        end_time = start_time + proc_time
        step_penalty = 0.0

        start_st, end_st = self.machine_to_stations[machine_id]
        waiting_start = self.job_ready_time[job_id]
        waiting_end = start_time

        if waiting_end > waiting_start:
            if self.check_capacity_violation(start_st, waiting_start, waiting_end):
                step_penalty -= 500.0
            self.station_records[start_st].append((waiting_start, waiting_end))

        self.section_records[machine_id].append((start_time, end_time, job_id))

        if is_curr_maintenance:
            self.section_records[opp_machine_id].append((start_time, end_time, job_id))

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

        reward = self._compute_reward() + step_penalty
        done = np.all(self.completed_jobs)

        return self.get_state(), reward, done, {
            'makespan': self.get_makespan(),
            'scheduled': (job_id, op_idx, machine_id, start_time, end_time)
        }