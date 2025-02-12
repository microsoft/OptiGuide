import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FacilityLocationTransportation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        # Randomly generate fixed costs for opening a facility
        fixed_costs = np.random.randint(self.min_cost, self.max_cost, self.number_of_facilities)

        # Randomly generate transportation costs between facilities and nodes
        transportation_costs = np.random.randint(self.min_cost, self.max_cost, (self.number_of_facilities, self.number_of_nodes))

        # Randomly generate capacities of facilities
        facility_capacities = np.random.randint(self.min_cap, self.max_cap, self.number_of_facilities)

        # Randomly generate demands for nodes
        node_demands = np.random.randint(self.min_demand, self.max_demand, self.number_of_nodes)
        
        # Randomly generate node-facility specific capacities
        node_facility_capacities = np.random.randint(self.node_facility_min_cap, self.node_facility_max_cap, (self.number_of_nodes, self.number_of_facilities))
        
        res = {
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'facility_capacities': facility_capacities,
            'node_demands': node_demands,
            'node_facility_capacities': node_facility_capacities,
        }
        
        ################## Incorporate FCMCNF Data Generation #################

        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        res['commodities'] = commodities
        res['adj_mat'] = adj_mat
        res['edge_list'] = edge_list
        res['incommings'] = incommings
        res['outcommings'] = outcommings
        
        ############### Additional Elements from Second MILP ###################

        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)
        compats = np.triu(np.random.rand(self.n_items, self.n_items), k=1)
        compats = compats + compats.transpose()
        compats = compats / compats.sum(1)

        bids = []
        n_dummy_items = 0

        while len(bids) < self.n_bids:
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            bidder_bids = {}

            prob = private_interests / private_interests.sum()
            item = np.random.choice(self.n_items, p=prob)
            bundle_mask = np.full(self.n_items, 0)
            bundle_mask[item] = 1

            while np.random.rand() < self.add_item_prob:
                if bundle_mask.sum() == self.n_items:
                    break
                item = self.choose_next_item(bundle_mask, private_interests, compats)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]
            price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)

            if price < 0:
                continue

            bidder_bids[frozenset(bundle)] = price

            sub_candidates = []
            for item in bundle:
                bundle_mask = np.full(self.n_items, 0)
                bundle_mask[item] = 1

                while bundle_mask.sum() < len(bundle):
                    item = self.choose_next_item(bundle_mask, private_interests, compats)
                    bundle_mask[item] = 1

                sub_bundle = np.nonzero(bundle_mask)[0]
                sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + self.additivity)
                sub_candidates.append((sub_bundle, sub_price))

            budget = self.budget_factor * price
            min_resale_value = self.resale_factor * values[bundle].sum()
            for bundle, price in [sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

                if len(bidder_bids) >= self.max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= self.n_bids:
                    break

                if price < 0 or price > budget:
                    continue

                if values[bundle].sum() < min_resale_value:
                    continue

                if frozenset(bundle) in bidder_bids:
                    continue

                bidder_bids[frozenset(bundle)] = price

            if len(bidder_bids) > 2:
                dummy_item = [self.n_items + n_dummy_items]
                n_dummy_items += 1
            else:
                dummy_item = []

            for bundle, price in bidder_bids.items():
                bids.append((list(bundle) + dummy_item, price))

        bids_per_item = [[] for item in range(self.n_items + n_dummy_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            bid1 = random.randint(0, len(bids) - 1)
            bid2 = random.randint(0, len(bids) - 1)
            if bid1 != bid2:
                mutual_exclusivity_pairs.append((bid1, bid2))

        high_priority_bids = random.sample(range(len(bids)), len(bids) // 3)

        res['bids'] = bids
        res['bids_per_item'] = bids_per_item
        res['mutual_exclusivity_pairs'] = mutual_exclusivity_pairs
        res['high_priority_bids'] = high_priority_bids
        
        return res

    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def choose_next_item(self, bundle_mask, interests, compats):
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return np.random.choice(len(interests), p=prob)

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        facility_capacities = instance['facility_capacities']
        node_demands = instance['node_demands']
        node_facility_capacities = instance['node_facility_capacities']
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        high_priority_bids = instance['high_priority_bids']

        number_of_facilities = len(fixed_costs)
        number_of_nodes = len(node_demands)
        number_of_bids = len(bids)
        
        M = 1e6  # Big M constant

        model = Model("FacilityLocationTransportationMulticommodityAuction")
       
        open_facility = {}
        transport_goods = {}
        node_demand_met = {}
        x_vars = {}
        y_vars = {}
        z_vars = {}
        bid_vars = {}
        patrol_vars = {}

        # Decision variables: y[j] = 1 if facility j is open
        for j in range(number_of_facilities):
            open_facility[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Decision variables: x[i][j] = amount of goods transported from facility j to node i
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                transport_goods[(i, j)] = model.addVar(vtype="C", name=f"x_{i}_{j}")

        # Decision variables: z[i] = 1 if demand of node i is met
        for i in range(number_of_nodes):
            node_demand_met[i] = model.addVar(vtype="B", name=f"z_{i}")

        # New variables for multicommodity flows
        for (i, j) in edge_list:
            for k in range(len(commodities)):
                x_vars[f"x_{i+1}_{j+1}_{k+1}"] = model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}")
                z_vars[f"z_{i+1}_{j+1}_{k+1}"] = model.addVar(vtype="C", name=f"z_{i+1}_{j+1}_{k+1}")
            y_vars[f"y_{i+1}_{j+1}"] = model.addVar(vtype="B", name=f"y_{i+1}_{j+1}")
        
        # New variables for combinatorial auction
        for i in range(number_of_bids):
            bid_vars[i] = model.addVar(vtype="B", name=f"Bid_{i}")

        # New variables for patrol allocation
        for bid in high_priority_bids:
            for j in range(number_of_facilities):
                patrol_vars[(bid, j)] = model.addVar(vtype="I", name=f"patrol_{bid}_{j}")

        # Objective: Minimize total cost with additional elements
        objective_expr = quicksum(fixed_costs[j] * open_facility[j] for j in range(number_of_facilities)) + \
                         quicksum(transportation_costs[j][i] * transport_goods[(i, j)] for i in range(number_of_nodes) for j in range(number_of_facilities))
        objective_expr += quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(len(commodities))
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr -= quicksum(
            bids[i][1] * bid_vars[i] for i in range(number_of_bids)
        )
        objective_expr += quicksum(
            patrol_vars[(bid, j)] for bid in high_priority_bids for j in range(number_of_facilities)
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints: Each node's demand must be met
        for i in range(number_of_nodes):
            model.addCons(
                quicksum(transport_goods[(i, j)] for j in range(number_of_facilities)) == node_demands[i],
                f"NodeDemand_{i}"
            )

        # Constraints: Facility capacity must not be exceeded 
        for j in range(number_of_facilities):
            model.addCons(
                quicksum(transport_goods[(i,j)] for i in range(number_of_nodes)) <= facility_capacities[j] * open_facility[j],
                f"FacilityCapacity_{j}"
            )

        # Constraints: Ensure transportation is feasible only if facility is open
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                model.addCons(
                    transport_goods[(i, j)] <= M * open_facility[j],
                    f"BigM_TransFeasibility_{i}_{j}"
                )
        
        # Constraints: Node-Facility specific capacity constraints
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                model.addCons(
                    transport_goods[(i, j)] <= node_facility_capacities[i][j],
                    f"NodeFacilityCap_{i}_{j}"
                )

        # Multicommodity Flow Constraints: Flow conservation for each commodity
        for i in range(self.n_nodes):
            for k in range(len(commodities)):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Multicommodity Arc Capacity Constraints and Complementarity Constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(len(commodities)))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

            for k in range(len(commodities)):
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] >= x_vars[f"x_{i+1}_{j+1}_{k+1}"] - (1 - y_vars[f"y_{i+1}_{j+1}"]) * adj_mat[i, j][2], f"comp1_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] <= x_vars[f"x_{i+1}_{j+1}_{k+1}"], f"comp2_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] <= y_vars[f"y_{i+1}_{j+1}"] * adj_mat[i, j][2], f"comp3_{i+1}_{j+1}_{k+1}")

        ### Additional Constraints from Second MILP

        # Each item is included in at most one winning bid
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f" Item_{item}")

        # Mutual exclusivity pairs of bids
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

        # Each bid should be allocated to a single facility
        for i in range(number_of_bids):
            for j in range(number_of_facilities):
                model.addCons(
                    bid_vars[i] <= open_facility[j],
                    f"BidFacility_{i}_{j}"
                )
        
        # Ensure high priority bids are covered by patrols
        for bid in high_priority_bids:
            for j in range(number_of_facilities):
                model.addCons(
                    patrol_vars[(bid, j)] >= bid_vars[bid],
                    f"HighPriorityBidPatrol_{bid}_{j}"
                )

        # Logical Constraint: If one bid is accepted, another must be as well
        if number_of_bids > 1:
            bid_A, bid_B = 0, 1  # Example bids
            model.addCons(
                bid_vars[bid_A] <= bid_vars[bid_B],
                "LogicalCondition_1"
            )

        # Logical Condition: Two bids cannot be accepted together
        if number_of_bids > 3:
            bid_C, bid_D = 2, 3  # Example bids
            model.addCons(
                bid_vars[bid_C] + bid_vars[bid_D] <= 1,
                "LogicalCondition_2"
            )

        # Logical Condition: Simultaneous acceptance required for another pair of bids
        if number_of_bids > 5:
            bid_E, bid_F = 4, 5  # Example bids
            model.addCons(
                bid_vars[bid_E] == bid_vars[bid_F],
                "LogicalCondition_3"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    parameters = {
        'number_of_facilities': 10,
        'number_of_nodes': 15,
        'min_cost': 50,
        'max_cost': 1000,
        'min_cap': 350,
        'max_cap': 1600,
        'min_demand': 60,
        'max_demand': 300,
        'node_facility_min_cap': 60,
        'node_facility_max_cap': 900,
        'min_n_nodes': 15,
        'max_n_nodes': 20,
        'min_n_commodities': 30,
        'max_n_commodities': 50,
        'c_range': (70, 350),
        'd_range': (200, 2000),
        'ratio': 500,
        'k_max': 60,
        'er_prob': 0.62,
        'n_items': 100,
        'n_bids': 50,
        'min_value': 50,
        'max_value': 200,
        'value_deviation': 0.1,
        'additivity': 0.1,
        'add_item_prob': 0.3,
        'budget_factor': 3,
        'resale_factor': 0.5,
        'max_n_sub_bids': 5,
        'n_exclusive_pairs': 20,
        'MaxPatrolBudget': 20000,
    }

    seed = 42
    facility_location = FacilityLocationTransportation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")