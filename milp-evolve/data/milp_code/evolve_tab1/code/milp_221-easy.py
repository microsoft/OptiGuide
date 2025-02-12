import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HRPA:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_patients = np.random.randint(self.min_patients, self.max_patients)
        G = nx.erdos_renyi_graph(n=n_patients, p=self.er_prob, directed=True, seed=self.seed)
        return G

    def generate_patients_resources_data(self, G):
        for node in G.nodes:
            G.nodes[node]['care_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['compatibility_cost'] = np.random.randint(1, 20)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def generate_incompatible_pairs(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E_invalid.add(edge)
        return E_invalid

    def find_patient_groups(self, G):
        cliques = list(nx.find_cliques(G.to_undirected()))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_patients_resources_data(G)
        E_invalid = self.generate_incompatible_pairs(G)
        groups = self.find_patient_groups(G)

        nurse_availability = {node: np.random.randint(1, 40) for node in G.nodes}
        coord_complexity = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}
        medication_compatibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

        def generate_bids_and_exclusivity(groups, n_exclusive_pairs):
            bids = [(group, np.random.uniform(50, 200)) for group in groups]
            mutual_exclusivity_pairs = set()
            while len(mutual_exclusivity_pairs) < n_exclusive_pairs:
                bid1 = np.random.randint(0, len(bids))
                bid2 = np.random.randint(0, len(bids))
                if bid1 != bid2:
                    mutual_exclusivity_pairs.add((bid1, bid2))
            return bids, list(mutual_exclusivity_pairs)

        bids, mutual_exclusivity_pairs = generate_bids_and_exclusivity(groups, self.n_exclusive_pairs)

        # New data for transportation and facilities
        n_transport_modes = 3
        transport_capacities = {mode: np.random.randint(5, 20) for mode in range(n_transport_modes)}
        transport_costs = {mode: np.random.uniform(50, 200) for mode in range(n_transport_modes)}
        facility_capacities = {facility: np.random.randint(50, 200) for facility in range(self.n_facilities)}
        geographic_regions = {node: np.random.randint(0, 5) for node in G.nodes}  # 5 regions as an example
        region_access_details = {region: np.random.uniform(0.5, 1.5) for region in range(5)}  # access modifier for equity

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'groups': groups, 
            'nurse_availability': nurse_availability, 
            'coord_complexity': coord_complexity,
            'medication_compatibility': medication_compatibility,
            'bids': bids,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
            'transport_capacities': transport_capacities,
            'transport_costs': transport_costs,
            'facility_capacities': facility_capacities,
            'geographic_regions': geographic_regions,
            'region_access_details': region_access_details,
        }
    
    def solve(self, instance):
        G, E_invalid, groups = instance['G'], instance['E_invalid'], instance['groups']
        nurse_availability = instance['nurse_availability']
        coord_complexity = instance['coord_complexity']
        medication_compatibility = instance['medication_compatibility']
        bids = instance['bids']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        transport_capacities = instance['transport_capacities']
        transport_costs = instance['transport_costs']
        facility_capacities = instance['facility_capacities']
        geographic_regions = instance['geographic_regions']
        region_access_details = instance['region_access_details']

        model = Model("HRPA")
        patient_vars = {f"p{node}": model.addVar(vtype="B", name=f"p{node}") for node in G.nodes}
        compat_vars = {f"m{u}_{v}": model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}
        weekly_budget = model.addVar(vtype="C", name="weekly_budget")

        # New bid variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}

        # New transportation variables
        transport_vars = {(mode, node): model.addVar(vtype="B", name=f"t{mode}_{node}") for mode in transport_capacities for node in G.nodes}

        # New facility variables
        facility_vars = {facility: model.addVar(vtype="C", name=f"Facility_{facility}") for facility in facility_capacities}

        objective_expr = quicksum(
            G.nodes[node]['care_demand'] * patient_vars[f"p{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['compatibility_cost'] * compat_vars[f"m{u}_{v}"]
            for u, v in E_invalid
        )

        objective_expr -= quicksum(
            nurse_availability[node] * G.nodes[node]['care_demand']
            for node in G.nodes
        )

        objective_expr -= quicksum(
            coord_complexity[(u, v)] * compat_vars[f"m{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            medication_compatibility[(u, v)] * compat_vars[f"m{u}_{v}"]
            for u, v in G.edges
        )

        # New objective component to maximize bid acceptance
        objective_expr += quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

        # New objective component to minimize transportation costs
        objective_expr -= quicksum(
            transport_costs[mode] * transport_vars[(mode, node)]
            for mode in transport_capacities for node in G.nodes
        )

        # New objective component for equitable access to regions
        objective_expr += quicksum(
            region_access_details[geographic_regions[node]] * patient_vars[f"p{node}"]
            for node in G.nodes
        )

        for i, group in enumerate(groups):
            model.addCons(
                quicksum(patient_vars[f"p{patient}"] for patient in group) <= 1,
                name=f"Group_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                patient_vars[f"p{u}"] + patient_vars[f"p{v}"] <= 1 + compat_vars[f"m{u}_{v}"],
                name=f"Comp1_{u}_{v}"
            )
            model.addCons(
                patient_vars[f"p{u}"] + patient_vars[f"p{v}"] >= 2 * compat_vars[f"m{u}_{v}"],
                name=f"Comp2_{u}_{v}"
            )

        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )

        # New constraints for bid mutual exclusivity
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

        # New constraints to match transportation capacities
        for mode in transport_capacities:
            model.addCons(quicksum(transport_vars[(mode, node)] for node in G.nodes) <= transport_capacities[mode], name=f"TransportCapacity_{mode}")

        # New constraints to ensure treatment facility capacities are not exceeded
        for facility in facility_capacities:
            model.addCons(facility_vars[facility] <= facility_capacities[facility], name=f"FacilityCapacity_{facility}")

        # New constraints ensuring patient allocation respects transport and facility capacity
        for node in G.nodes:
            model.addCons(
                quicksum(transport_vars[(mode, node)] for mode in transport_capacities) >= patient_vars[f"p{node}"],
                name=f"TransportAssignment_{node}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_patients': 30,
        'max_patients': 411,
        'er_prob': 0.24,
        'alpha': 0.59,
        'weekly_budget_limit': 560,
        'n_exclusive_pairs': 2,
        'n_facilities': 5,
    }

    hrpa = HRPA(parameters, seed=seed)
    instance = hrpa.generate_instance()
    solve_status, solve_time = hrpa.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")