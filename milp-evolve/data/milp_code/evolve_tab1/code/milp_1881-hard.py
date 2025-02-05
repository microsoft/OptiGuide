import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class RenewableEnergySupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.num_plants > 0 and self.num_centers > 0
        assert self.min_gen_cost >= 0 and self.max_gen_cost >= self.min_gen_cost
        assert self.min_transmission_cost >= 0 and self.max_transmission_cost >= self.min_transmission_cost
        assert self.min_energy_capacity > 0 and self.max_energy_capacity >= self.min_energy_capacity
        assert self.min_exp_cost >= 0 and self.max_exp_cost >= self.min_exp_cost
        assert self.min_lambda_cost >= 0 and self.max_lambda_cost >= self.min_lambda_cost
        assert self.min_lambda_capacity > 0 and self.max_lambda_capacity >= self.min_lambda_capacity

        # Generate costs and capacities
        gen_costs = np.random.gamma(2, 2, self.num_plants) * (self.max_gen_cost - self.min_gen_cost) / 4 + self.min_gen_cost
        transmission_costs = np.random.gamma(2, 2, (self.num_plants, self.num_centers)) * (self.max_transmission_cost - self.min_transmission_cost) / 4 + self.min_transmission_cost
        energy_capacities = np.random.randint(self.min_energy_capacity, self.max_energy_capacity + 1, self.num_plants)
        center_demands = np.random.randint(1, 20, self.num_centers)
        distances = np.random.uniform(0, self.max_transmission_distance, (self.num_plants, self.num_centers))

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.num_plants):
            for d in range(self.num_centers):
                G.add_edge(f"plant_{p}", f"center_{d}")
                node_pairs.append((f"plant_{p}", f"center_{d}"))

        energy_types = 3
        type_center_demands = np.random.randint(1, 20, (energy_types, self.num_centers))

        expansion_costs = np.random.randint(self.min_exp_cost, self.max_exp_cost + 1, self.num_plants)
        transmission_setup_costs = np.random.randint(self.min_lambda_cost, self.max_lambda_cost + 1, (self.num_plants, self.num_centers))
        exp_capacities = np.random.randint(self.min_lambda_capacity, self.max_lambda_capacity + 1, self.num_plants)
        flow_capacities = np.random.randint(1, 100, (self.num_plants, self.num_centers))

        # Generating cliques for clique inequalities
        cliques = list(nx.find_cliques(G.to_undirected()))

        return {
            "gen_costs": gen_costs,
            "transmission_costs": transmission_costs,
            "energy_capacities": energy_capacities,
            "center_demands": center_demands,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "energy_types": energy_types,
            "type_center_demands": type_center_demands,
            "expansion_costs": expansion_costs,
            "transmission_setup_costs": transmission_setup_costs,
            "exp_capacities": exp_capacities,
            "flow_capacities": flow_capacities,
            "cliques": cliques,
        }

    def solve(self, instance):
        gen_costs = instance['gen_costs']
        transmission_costs = instance['transmission_costs']
        energy_capacities = instance['energy_capacities']
        center_demands = instance['center_demands']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        energy_types = instance['energy_types']
        type_center_demands = instance['type_center_demands']
        expansion_costs = instance['expansion_costs']
        transmission_setup_costs = instance['transmission_setup_costs']
        exp_capacities = instance['exp_capacities']
        flow_capacities = instance['flow_capacities']
        cliques = instance['cliques']

        model = Model("RenewableEnergySupplyChainOptimization")
        num_plants = len(gen_costs)
        num_centers = len(transmission_costs[0])

        # Decision variables
        gen_vars = {p: model.addVar(vtype="B", name=f"Plant_{p}") for p in range(num_plants)}
        transmission_vars = {k: model.addVar(vtype="I", name=f"Transmission_{k[0]}_{k[1]}") for k in node_pairs}
        energy_type_vars = {(etype, u, v): model.addVar(vtype="C", name=f"Energy_{etype}_{u}_{v}") for etype in range(energy_types) for u, v in node_pairs}
        expansion_vars = {p: model.addVar(vtype="I", name=f"Expansion_{p}") for p in range(num_plants)}
        setup_vars = {(p, d): model.addVar(vtype="B", name=f"Setup_{p}_{d}") for p in range(num_plants) for d in range(num_centers)}
        flow_vars = {(p, d): model.addVar(vtype="I", name=f"Flow_{p}_{d}") for p in range(num_plants) for d in range(num_centers)}
        clique_vars = {i: model.addVar(vtype="B", name=f"Clique_{i}") for i in range(len(cliques))}

        # Objective function
        model.setObjective(
            quicksum(gen_costs[p] * gen_vars[p] for p in range(num_plants)) +
            quicksum(transmission_costs[int(u.split('_')[1]), int(v.split('_')[1])] * transmission_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(expansion_costs[p] * expansion_vars[p] for p in range(num_plants)) +
            quicksum(transmission_setup_costs[p, d] * setup_vars[p, d] for p in range(num_plants) for d in range(num_centers)) +
            quicksum(flow_vars[(p, d)] / flow_capacities[p, d] for p in range(num_plants) for d in range(num_centers)), 
            "minimize"
        )

        # Constraints
        for d in range(num_centers):
            for etype in range(energy_types):
                model.addCons(
                    quicksum(energy_type_vars[(etype, u, f"center_{d}")] for u in G.predecessors(f"center_{d}")) >= type_center_demands[etype, d], 
                    f"Center_{d}_EnergyTypeConservation_{etype}"
                )

        for p in range(num_plants):
            model.addCons(
                quicksum(energy_type_vars[(etype, f"plant_{p}", f"center_{d}")] for d in range(num_centers) for etype in range(energy_types)) <= energy_capacities[p], 
                f"Plant_{p}_MaxEnergyCapacity"
            )

        for d in range(num_centers):
            model.addCons(
                quicksum(gen_vars[p] for p in range(num_plants) if distances[p, d] <= self.max_transmission_distance) >= 1, 
                f"Center_{d}_RegionCoverage"
            )

        for u, v in node_pairs:
            model.addCons(
                transmission_vars[(u, v)] <= flow_capacities[int(u.split('_')[1]), int(v.split('_')[1])], 
                f"TransmissionCapacity_{u}_{v}"
            )

        for d in range(num_centers):
            model.addCons(
                quicksum(setup_vars[p, d] for p in range(num_plants)) == 1, 
                f"Center_{d}_SetupAssignment"
            )

        for p in range(num_plants):
            for d in range(num_centers):
                model.addCons(
                    setup_vars[p, d] <= expansion_vars[p], 
                    f"Plant_{p}_Setup_{d}"
                )

        for p in range(num_plants):
            model.addCons(
                quicksum(center_demands[d] * setup_vars[p, d] for d in range(num_centers)) <= exp_capacities[p] * expansion_vars[p], 
                f"Plant_{p}_ExpansionCapacity"
            )
        
        for p in range(num_plants):
            model.addCons(
                quicksum(flow_vars[(p, d)] for d in range(num_centers)) <= energy_capacities[p], 
                f"Plant_{p}_FlowConservation"
            )
        
        for d in range(num_centers):
            model.addCons(
                quicksum(flow_vars[(p, d)] for p in range(num_plants)) == center_demands[d], 
                f"Center_{d}_FlowSatisfaction"
            )
        
        for p in range(num_plants):
            for d in range(num_centers):
                model.addCons(
                    flow_vars[(p, d)] <= flow_capacities[p, d], 
                    f"FlowCapacity_{p}_{d}"
                )

        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(expansion_vars[int(node.split('_')[1])] for node in clique) <= len(clique) * clique_vars[i], 
                f"CliqueActivation_{i}"
            )
            model.addCons(
                clique_vars[i] <= 1, 
                f"CliqueSingleActivation_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_plants': 250,
        'num_centers': 50,
        'min_transmission_cost': 800,
        'max_transmission_cost': 1200,
        'min_gen_cost': 500,
        'max_gen_cost': 1200,
        'min_energy_capacity': 2000,
        'max_energy_capacity': 5000,
        'max_transmission_distance': 800,
        'min_exp_cost': 2000,
        'max_exp_cost': 7000,
        'min_lambda_cost': 3000,
        'max_lambda_cost': 5000,
        'min_lambda_capacity': 1800,
        'max_lambda_capacity': 3000,
    }

    optimizer = RenewableEnergySupplyChainOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")