import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ResourceAllocationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.min_task_value >= 0 and self.max_task_value >= self.min_task_value
        assert self.add_task_prob >= 0 and self.add_task_prob <= 1

        def choose_next_task(bundle_mask, interests, compats):
            n_tasks = len(interests)
            prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
            prob /= prob.sum()
            return np.random.choice(n_tasks, p=prob)

        values = self.min_task_value + (self.max_task_value - self.min_task_value) * np.random.rand(self.n_tasks)
        compats = np.triu(np.random.rand(self.n_tasks, self.n_tasks), k=1)
        compats = compats + compats.transpose()
        compats = compats / compats.sum(1)

        tasks = []
        n_extra_items = 0

        while len(tasks) < self.n_tasks:
            interest_level = np.random.rand(self.n_tasks)
            private_values = values + self.max_task_value * self.value_deviation * (2 * interest_level - 1)

            task_bids = {}

            prob = interest_level / interest_level.sum()
            item = np.random.choice(self.n_tasks, p=prob)
            bundle_mask = np.full(self.n_tasks, 0)
            bundle_mask[item] = 1

            while np.random.rand() < self.add_task_prob:
                if bundle_mask.sum() == self.n_tasks:
                    break
                item = choose_next_task(bundle_mask, interest_level, compats)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]
            total_value = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            if total_value < 0:
                continue

            task_bids[frozenset(bundle)] = total_value

            sub_candidates = []
            for item in bundle:
                bundle_mask = np.full(self.n_tasks, 0)
                bundle_mask[item] = 1

                while bundle_mask.sum() < len(bundle):
                    item = choose_next_task(bundle_mask, interest_level, compats)
                    bundle_mask[item] = 1

                sub_bundle = np.nonzero(bundle_mask)[0]
                sub_value = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)
                sub_candidates.append((sub_bundle, sub_value))

            budget = self.budget_factor * total_value
            min_resale_value = self.resale_factor * values[bundle].sum()
            for bundle, value in [
                sub_candidates[i] for i in np.argsort([-value for bundle, value in sub_candidates])]:

                if len(task_bids) >= self.max_sub_bids + 1 or len(tasks) + len(task_bids) >= self.n_tasks:
                    break

                if value < 0 or value > budget:
                    continue

                if values[bundle].sum() < min_resale_value:
                    continue

                if frozenset(bundle) in task_bids:
                    continue

                task_bids[frozenset(bundle)] = value

            if len(task_bids) > 2:
                extra_item = [self.n_tasks + n_extra_items]
                n_extra_items += 1
            else:
                extra_item = []

            for bundle, value in task_bids.items():
                tasks.append((list(bundle) + extra_item, value))

        tasks_per_item = [[] for item in range(self.n_tasks + n_extra_items)]
        for i, task in enumerate(tasks):
            bundle, value = task
            for item in bundle:
                tasks_per_item[item].append(i)

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            task1 = random.randint(0, len(tasks) - 1)
            task2 = random.randint(0, len(tasks) - 1)
            if task1 != task2:
                mutual_exclusivity_pairs.append((task1, task2))

        # Facility data generation
        n_production_lines = np.random.randint(self.line_min_count, self.line_max_count)
        operational_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_production_lines).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(tasks)).tolist()
        capacity = np.random.randint(10, 50, size=n_production_lines).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_production_lines).tolist()
        throughput = np.random.uniform(1.0, 5.0, size=len(tasks)).tolist()

        return {
            "tasks": tasks,
            "tasks_per_item": tasks_per_item,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "n_production_lines": n_production_lines,
            "operational_cost": operational_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "throughput": throughput
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        tasks = instance['tasks']
        tasks_per_item = instance['tasks_per_item']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        n_production_lines = instance['n_production_lines']
        operational_cost = instance['operational_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        throughput = instance['throughput']

        model = Model("ResourceAllocationOptimization")

        task_vars = {i: model.addVar(vtype="B", name=f"Task_{i}") for i in range(len(tasks))}
        open_line = {j: model.addVar(vtype="B", name=f"OpenLine_{j}") for j in range(n_production_lines)}
        task_to_line = {(i, j): model.addVar(vtype="B", name=f"TaskLine_{i}_{j}") for i in range(len(tasks)) for j in range(n_production_lines)}
        line_throughput = {i: model.addVar(vtype="C", name=f"Throughput_{i}") for i in range(len(tasks))}

        objective_expr = quicksum(value * task_vars[i] for i, (bundle, value) in enumerate(tasks)) \
                         - quicksum(operational_cost[j] * open_line[j] for j in range(n_production_lines)) \
                         - quicksum(assignment_cost[i] * quicksum(task_to_line[i, j] for j in range(n_production_lines)) for i in range(len(tasks))) \
                         - quicksum(setup_cost[j] * open_line[j] for j in range(n_production_lines))

        # Constraints: Each item can only be part of one accepted task
        for item, task_indices in enumerate(tasks_per_item):
            model.addCons(quicksum(task_vars[task_idx] for task_idx in task_indices) <= 1, f"Item_{item}")

        # Mutually exclusive task pairs using Big M Constraint
        for (task1, task2) in mutual_exclusivity_pairs:
            M = 1  # Since binary variables can only be 0 or 1
            model.addCons(task_vars[task1] + task_vars[task2] <= M, f"Exclusive_{task1}_{task2}")

        # Task assignment to production line
        for i in range(len(tasks)):
            model.addCons(quicksum(task_to_line[i, j] for j in range(n_production_lines)) == task_vars[i], f"TaskLine_{i}")

        # Line capacity constraints
        for j in range(n_production_lines):
            model.addCons(quicksum(task_to_line[i, j] for i in range(len(tasks))) <= capacity[j] * open_line[j], f"LineCapacity_{j}")

        # Throughput constraints
        for i in range(len(tasks)):
            model.addCons(line_throughput[i] == quicksum(throughput[i] * task_to_line[i, j] for j in range(n_production_lines)), f"Throughput_{i}")

        max_throughput = np.max(throughput) * len(tasks)
        for j in range(n_production_lines):
            model.addCons(quicksum(line_throughput[i] * task_to_line[i, j] for i in range(len(tasks))) <= max_throughput * open_line[j], f"MaxThroughput_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_tasks': 100,
        'min_task_value': 2,
        'max_task_value': 5000,
        'value_deviation': 0.17,
        'additivity': 0.31,
        'add_task_prob': 0.1,
        'budget_factor': 1350.0,
        'resale_factor': 0.31,
        'max_sub_bids': 135,
        'n_exclusive_pairs': 240,
        'line_min_count': 0,
        'line_max_count': 131,
    }

    optimization = ResourceAllocationOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")