import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_network_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.edge_prob, directed=True, seed=self.seed)
        return G
    
    def generate_supply_chain_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(50, 300)
            G.nodes[node]['supply'] = np.random.randint(100, 500)
        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.uniform(0.5, 5.0)
            G[u][v]['capacity'] = np.random.randint(20, 100)

    def create_contracts(self, G):
        contracts = list(nx.find_cliques(G.to_undirected()))
        return contracts
    
    def get_instance(self):
        G = self.generate_network_graph()
        self.generate_supply_chain_data(G)
        contracts = self.create_contracts(G)
        
        warehouse_cap = {node: np.random.randint(150, 500) for node in G.nodes}
        displacement_cost = {(u, v): np.random.uniform(2.0, 10.0) for u, v in G.edges}
        monthly_contracts = [(contract, np.random.uniform(1000, 5000)) for contract in contracts]

        transportation_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            transportation_scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['demand'], G.nodes[node]['demand'] * self.demand_variation)
                                                    for node in G.nodes}
            transportation_scenarios[s]['transport_cost'] = {(u, v): np.random.normal(G[u][v]['transport_cost'], G[u][v]['transport_cost'] * self.cost_variation)
                                                            for u, v in G.edges}
            transportation_scenarios[s]['warehouse_cap'] = {node: np.random.normal(warehouse_cap[node], warehouse_cap[node] * self.cap_variation)
                                                           for node in G.nodes}
        
        revenue_rewards = {node: np.random.uniform(100, 1000) for node in G.nodes}

        return {
            'G': G,
            'contracts': contracts,
            'warehouse_cap': warehouse_cap,
            'displacement_cost': displacement_cost,
            'monthly_contracts': monthly_contracts,
            'transportation_scenarios': transportation_scenarios,
            'revenue_rewards': revenue_rewards,
        }

    def solve(self, instance):
        G, contracts = instance['G'], instance['contracts']
        warehouse_cap = instance['warehouse_cap']
        displacement_cost = instance['displacement_cost']
        monthly_contracts = instance['monthly_contracts']
        transportation_scenarios = instance['transportation_scenarios']
        revenue_rewards = instance['revenue_rewards']

        model = Model("SupplyChainOptimization")
        
        # Define variables
        supply_shift_vars = {f"NewSupply_{node}": model.addVar(vtype="B", name=f"NewSupply_{node}") for node in G.nodes}
        supply_transport_vars = {f"Transport_{u}_{v}": model.addVar(vtype="B", name=f"Transport_{u}_{v}") for u, v in G.edges}
        monthly_contract_vars = {i: model.addVar(vtype="B", name=f"ContractHigh_{i}") for i in range(len(monthly_contracts))}
        
        # Objective function
        objective_expr = quicksum(
            transportation_scenarios[s]['demand'][node] * supply_shift_vars[f"NewSupply_{node}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            displacement_cost[(u, v)] * supply_transport_vars[f"Transport_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * monthly_contract_vars[i] for i, (contract, price) in enumerate(monthly_contracts))
        objective_expr += quicksum(revenue_rewards[node] * supply_shift_vars[f"NewSupply_{node}"] for node in G.nodes)
        
        # Constraints
        for i, contract in enumerate(contracts):
            model.addCons(
                quicksum(supply_shift_vars[f"NewSupply_{node}"] for node in contract) <= 1,
                name=f"Contract_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                supply_shift_vars[f"NewSupply_{u}"] + supply_shift_vars[f"NewSupply_{v}"] <= 1 + supply_transport_vars[f"Transport_{u}_{v}"],
                name=f"SupplyTransport_{u}_{v}"
            )
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 21,
        'max_nodes': 247,
        'edge_prob': 0.17,
        'storage_hours': 2700,
        'no_of_scenarios': 25,
        'demand_variation': 0.52,
        'cost_variation': 0.73,
        'cap_variation': 0.1,
    }

    supply_chain = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_chain.get_instance()
    solve_status, solve_time = supply_chain.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")