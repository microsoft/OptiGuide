import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SetCoverModified:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        ################# Set Cover Instance #################
        nnzrs = int(self.n_rows * self.n_cols * self.density)

        # compute number of rows per column
        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at least 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        # for each column, sample random rows
        indices[:self.n_rows] = np.random.permutation(self.n_rows)  # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)
            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)

            i += n
            indptr.append(i)

        # objective coefficients
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        set_cover_instance = {'c': c, 'indptr_csr': indptr_csr, 'indices_csr': indices_csr}

        ################# Community Centers Instance #################
        n_nodes = np.random.randint(self.min_zones, self.max_zones)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.market_connectivity, seed=self.seed)

        for node in G.nodes:
            G.nodes[node]['population'] = np.random.randint(500, 5000)
            G.nodes[node]['temperature'] = np.random.randint(20, 100)
            G.nodes[node]['hazard_weight'] = np.random.uniform(1, 10)
        for u, v in G.edges:
            G[u][v]['travel_time'] = np.random.randint(5, 50)
            G[u][v]['capacity'] = np.random.randint(50, 200)
            G[u][v]['hazard_cost'] = (G.nodes[u]['hazard_weight'] + G.nodes[v]['hazard_weight']) / float(self.temp_param)

        neighborhoods = list(nx.find_cliques(G))
        center_capacity = {node: np.random.randint(200, 1000) for node in G.nodes}

        operation_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            operation_scenarios[s]['population'] = {node: np.random.normal(G.nodes[node]['population'], G.nodes[node]['population'] * self.population_variation)
                                                    for node in G.nodes}
            operation_scenarios[s]['travel_time'] = {(u, v): np.random.normal(G[u][v]['travel_time'], G[u][v]['travel_time'] * self.time_variation)
                                                     for u, v in G.edges}
            operation_scenarios[s]['center_capacity'] = {node: np.random.normal(center_capacity[node], center_capacity[node] * self.capacity_variation)
                                                         for node in G.nodes}
        service_revenue = {node: np.random.uniform(100, 400) for node in G.nodes}
        operational_costs = {(u, v): np.random.uniform(20.0, 60.0) for u, v in G.edges}

        n_centers = np.random.randint(self.center_min_count, self.center_max_count)
        comm_distance_matrix = np.random.uniform(5, 500, size=(n_centers, n_centers)).tolist()
        population_distribution = np.random.dirichlet(np.ones(3), size=n_centers).tolist()
        regional_costs = np.random.normal(loc=300, scale=50, size=n_centers).tolist()
        utility_costs = np.random.normal(loc=300, scale=50, size=n_centers).tolist()
        setup_cost = np.random.uniform(200, 1000, size=n_centers).tolist()
        service_flow_costs = {(u, v): np.random.uniform(2, 30) for u, v in G.edges}

        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E2.add(edge)

        community_centers_instance = {
            'G': G,
            'neighborhoods': neighborhoods,
            'center_capacity': center_capacity,
            'operation_scenarios': operation_scenarios,
            'service_revenue': service_revenue,
            'operational_costs': operational_costs,
            'n_centers': n_centers,
            'comm_distance_matrix': comm_distance_matrix,
            'population_distribution': population_distribution,
            'regional_costs': regional_costs,
            'utility_costs': utility_costs,
            'setup_cost': setup_cost,
            'service_flow_costs': service_flow_costs,
            'E2': E2
        }

        return set_cover_instance, community_centers_instance

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        set_cover_instance, community_centers_instance = instance
        c = set_cover_instance['c']
        indptr_csr = set_cover_instance['indptr_csr']
        indices_csr = set_cover_instance['indices_csr']

        model = Model("SetCoverModified")
        var_names = {}

        # Create Set Cover variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        G, neighborhoods = community_centers_instance['G'], community_centers_instance['neighborhoods']
        center_capacity = community_centers_instance['center_capacity']
        operation_scenarios = community_centers_instance['operation_scenarios']
        service_revenue = community_centers_instance['service_revenue']
        operational_costs = community_centers_instance['operational_costs']
        n_centers = community_centers_instance['n_centers']
        comm_distance_matrix = community_centers_instance['comm_distance_matrix']
        population_distribution = community_centers_instance['population_distribution']
        regional_costs = community_centers_instance['regional_costs']
        utility_costs = community_centers_instance['utility_costs']
        setup_cost = community_centers_instance['setup_cost']
        service_flow_costs = community_centers_instance['service_flow_costs']
        E2 = community_centers_instance['E2']

        zone_vars = {node: model.addVar(vtype="B", name=f"Zone{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        traffic_var = model.addVar(vtype="C", name="traffic_var")
        daily_traffic_vars = {i: model.addVar(vtype="B", name=f"Traffic_{i}") for i in range(len(neighborhoods))}
        segment_vars = {(u, v): {segment: model.addVar(vtype="B", name=f"Segment_{u}_{v}_{segment}") for segment in range(1, self.piecewise_segments + 1)} for u, v in G.edges}

        utility_use_vars = {node: model.addVar(vtype="B", name=f"UtilityUse{node}") for node in G.nodes}
        community_vars = {node: model.addVar(vtype="I", name=f"Community{node}", lb=1, ub=5) for node in G.nodes}
        center_vars = {j: model.addVar(vtype="B", name=f"Center{j}") for j in range(n_centers)}
        travel_cost_vars = {j: model.addVar(vtype="C", name=f"CommTravelCost{j}") for j in range(n_centers)}
        demographic_vars = {(g, j): model.addVar(vtype="B", name=f"Demographic_{g}_{j}") for g in range(3) for j in range(n_centers)}
        regional_cost_vars = {j: model.addVar(vtype="C", name=f"RegionalCost{j}") for j in range(n_centers)}
        service_flow_vars = {f"ServiceFlow_{u}_{v}": model.addVar(vtype="C", name=f"ServiceFlow_{u}_{v}", lb=0) for u, v in G.edges}
        cooling_node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        hazard_edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Objective function
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        
        # Include the Community Centers objective components
        objective_expr += quicksum(
            operation_scenarios[s]['population'][node] * zone_vars[node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            quicksum(segment * segment_vars[(u,v)][segment] for segment in range(1, self.piecewise_segments + 1))
            for u, v in G.edges
        )
        objective_expr += quicksum(service_revenue[node] * zone_vars[node] for node in G.nodes)
        objective_expr -= quicksum(operational_costs[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G.edges)
        objective_expr -= quicksum(setup_cost[j] * center_vars[j] for j in range(n_centers))
        objective_expr -= quicksum(travel_cost_vars[j] for j in range(n_centers))
        objective_expr -= quicksum(regional_cost_vars[j] for j in range(n_centers))
        objective_expr -= quicksum(service_flow_costs[(u,v)] * service_flow_vars[f"ServiceFlow_{u}_{v}"] for u, v in G.edges)
        objective_expr += quicksum(G.nodes[node]['temperature'] * cooling_node_vars[f"x{node}"] for node in G.nodes)
        objective_expr -= quicksum(G[u][v]['hazard_cost'] * hazard_edge_vars[f"y{u}_{v}"] for u, v in E2)

        # Set Objective
        model.setObjective(objective_expr, "minimize")

        ################# Adding Constraints #################
        # Community Centers Constraints
        for node in G.nodes:
            model.addCons(utility_use_vars[node] <= zone_vars[node], name=f"UtilityUseConstraint_{node}")
            model.addCons(quicksum(segment_vars[(u, v)][segment] for u, v in G.edges if u == node or v == node for segment in range(1, self.piecewise_segments + 1)) <= community_vars[node] * 15, name=f"CommunityConstraint_{node}")

        for j in range(n_centers):
            model.addCons(quicksum(comm_distance_matrix[j][k] * center_vars[k] for k in range(n_centers)) == travel_cost_vars[j], f"CommTravelCost_{j}")

        for g in range(3):
            for j in range(n_centers):
                model.addCons(demographic_vars[(g, j)] <= population_distribution[j][g] * center_vars[j], f"Demographic_{g}_{j}")

        for j in range(n_centers):
            model.addCons(regional_cost_vars[j] == regional_costs[j] * center_vars[j], f"RegionalCost_{j}")

        for i, neighborhood in enumerate(neighborhoods):
            model.addCons(quicksum(zone_vars[node] for node in neighborhood) <= 1, name=f"ZoneGroup_{i}")

        for u, v in G.edges:
            model.addCons(zone_vars[u] + zone_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"], name=f"ServiceFlow_{u}_{v}")
            model.addCons(route_vars[f"Route_{u}_{v}"] == quicksum(segment_vars[(u, v)][segment] for segment in range(1, self.piecewise_segments + 1)), name=f"PiecewiseDist_{u}_{v}")
            model.addCons(service_flow_vars[f"ServiceFlow_{u}_{v}"] <= G[u][v]['capacity'], name=f"Capacity_{u}_{v}")

        model.addCons(traffic_var <= self.service_hours, name="ServiceTime_Limit")

        for node in G.nodes:
            inflow = quicksum(service_flow_vars[f"ServiceFlow_{u}_{v}"] for u, v in G.edges if v == node)
            outflow = quicksum(service_flow_vars[f"ServiceFlow_{u}_{v}"] for u, v in G.edges if u == node)
            model.addCons(inflow == outflow, name=f"ServiceFlow_Continuity_{node}")

        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(cooling_node_vars[f"x{u}"] + cooling_node_vars[f"x{v}"] - hazard_edge_vars[f"y{u}_{v}"] <= 1, name=f"C_hazard_{u}_{v}")
            else:
                model.addCons(cooling_node_vars[f"x{u}"] + cooling_node_vars[f"x{v}"] <= 1, name=f"C_safe_{u}_{v}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 562,
        'n_cols': 750,
        'density': 0.17,
        'max_coef': 25,
        'min_zones': 27,
        'max_zones': 250,
        'market_connectivity': 10,
        'service_hours': 1012,
        'piecewise_segments': 5,
        'no_of_scenarios': 300,
        'population_variation': 0.66,
        'time_variation': 0.17,
        'capacity_variation': 0.73,
        'center_min_count': 11,
        'center_max_count': 187,
        'temp_param': 0.94,
        'beta': 0.8,
    }
    set_cover_modified_problem = SetCoverModified(parameters, seed=seed)
    instance = set_cover_modified_problem.generate_instance()
    solve_status, solve_time = set_cover_modified_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")