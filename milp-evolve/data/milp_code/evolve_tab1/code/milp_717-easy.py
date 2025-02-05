import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class EmergencyServiceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_instance(self):
        graph = nx.barabasi_albert_graph(self.num_nodes, self.num_edges)
        demand_nodes = list(graph.nodes)[:self.num_demand_nodes]
        service_nodes = list(graph.nodes)[:self.num_service_nodes]
        
        demands = self.randint(self.num_demand_nodes, self.demand_interval)
        capacities = self.randint(self.num_service_nodes, self.capacity_interval)
        fixed_costs = self.randint(self.num_service_nodes, self.fixed_cost_interval)
        maintenance_costs = self.randint(self.num_service_nodes, self.maintenance_cost_interval)

        edge_weights = {(u, v): np.random.rand()*10 for u, v in graph.edges}
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'maintenance_costs': maintenance_costs,
            'graph': graph,
            'edge_weights': edge_weights,
            'demand_nodes': demand_nodes,
            'service_nodes': service_nodes
        }

        reliability_coeff = 100 * np.random.rand()
        res.update({"reliability_coeff": reliability_coeff})

        return res

    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        maintenance_costs = instance['maintenance_costs']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        service_nodes = instance['service_nodes']
        demand_nodes = instance['demand_nodes']
        reliability_coeff = instance['reliability_coeff']

        num_demand_nodes = len(demand_nodes)
        num_service_nodes = len(service_nodes)
        
        model = Model("EmergencyServiceOptimization")
        
        node_operation = {j: model.addVar(vtype="B", name=f"NodeOperation_{j}") for j in service_nodes}
        service_flow = {(i, j): model.addVar(vtype="B", name=f"ServiceFlow_{i}_{j}") for i in demand_nodes for j in service_nodes}
        reliability = {j: model.addVar(vtype="I", name=f"Reliability_{j}") for j in service_nodes}
        high_maintenance_cost = {j: model.addVar(vtype="C", name=f"HighMaintenanceCost_{j}") for j in service_nodes}

        # Primary objective: Minimize response time and maximize reliability
        objective_expr = quicksum(fixed_costs[j] * node_operation[j] for j in service_nodes) + \
                         quicksum(edge_weights[u, v] * service_flow[i, j] for i in demand_nodes for j in service_nodes
                                  for u, v in graph.edges if u == i and v == j) + \
                         quicksum(reliability_coeff * reliability[j] for j in service_nodes) + \
                         quicksum(maintenance_costs[j] * high_maintenance_cost[j] for j in service_nodes)

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each demand node must be served by at least one service node
        for i in demand_nodes:
            model.addCons(quicksum(service_flow[i, j] for j in service_nodes) >= 1, f"CustomerFlow_{i}")
        
        # Constraint: Capacity constraints
        for j in service_nodes:
            model.addCons(quicksum(service_flow[i, j] * demands[i] for i in demand_nodes) <= capacities[j] * node_operation[j], f"Capacity_{j}")

        # Constraint: Total operational cost constraints
        total_cost = np.sum(fixed_costs)
        model.addCons(quicksum(fixed_costs[j] * node_operation[j] for j in service_nodes) <= total_cost, "MaximumCost")

        # Constraint: Assigning high maintenance costs for nodes with high operational load
        high_cost_threshold = np.percentile(fixed_costs, 90)
        for j in service_nodes:
            if fixed_costs[j] > high_cost_threshold:
                model.addCons(high_maintenance_cost[j] >= fixed_costs[j], f"HighMaintenance_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_nodes': 350,
        'num_edges': 27,
        'num_demand_nodes': 225,
        'num_service_nodes': 80,
        'demand_interval': (35, 140),
        'capacity_interval': (500, 2000),
        'fixed_cost_interval': (100, 500),
        'maintenance_cost_interval': (7, 70),
    }

    emergency_service_optimization = EmergencyServiceOptimization(parameters, seed=seed)
    instance = emergency_service_optimization.generate_instance()
    solve_status, solve_time = emergency_service_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")