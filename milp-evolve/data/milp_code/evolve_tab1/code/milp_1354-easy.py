import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class UrbanPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.Number_of_Zones > 0 and self.Number_of_Projects > 0
        assert self.Min_Zone_Cost >= 0 and self.Max_Zone_Cost >= self.Min_Zone_Cost
        assert self.Project_Cost_Lower_Bound >= 0 and self.Project_Cost_Upper_Bound >= self.Project_Cost_Lower_Bound
        assert self.Min_Zone_Capacity > 0 and self.Max_Zone_Capacity >= self.Min_Zone_Capacity

        zone_costs = np.random.randint(self.Min_Zone_Cost, self.Max_Zone_Cost + 1, self.Number_of_Zones)
        project_costs = np.random.randint(self.Project_Cost_Lower_Bound, self.Project_Cost_Upper_Bound + 1, (self.Number_of_Zones, self.Number_of_Projects))
        maintenance_costs = np.random.randint(self.Min_Zone_Capacity, self.Max_Zone_Capacity + 1, self.Number_of_Zones)
        project_requirements = np.random.randint(1, 10, self.Number_of_Projects)
        impact_values = np.random.uniform(0.8, 1.0, (self.Number_of_Zones, self.Number_of_Projects))
        
        planning_scenarios = [{} for _ in range(self.No_of_Scenarios)]
        for s in range(self.No_of_Scenarios):
            planning_scenarios[s]['requirement'] = {p: max(0, np.random.gamma(project_requirements[p], project_requirements[p] * self.Requirement_Variation)) for p in range(self.Number_of_Projects)}

        return {
            "zone_costs": zone_costs,
            "project_costs": project_costs,
            "maintenance_costs": maintenance_costs,
            "project_requirements": project_requirements,
            "impact_values": impact_values,
            "planning_scenarios": planning_scenarios
        }
        
    def solve(self, instance):
        zone_costs = instance['zone_costs']
        project_costs = instance['project_costs']
        maintenance_costs = instance['maintenance_costs']
        planning_scenarios = instance['planning_scenarios']
        impact_values = instance['impact_values']
        
        model = Model("UrbanPlanning")
        number_of_zones = len(zone_costs)
        number_of_projects = len(project_costs[0])
        no_of_scenarios = len(planning_scenarios)

        # Decision variables
        zone_vars = {z: model.addVar(vtype="B", name=f"Zone_{z}") for z in range(number_of_zones)}
        project_vars = {(z, p): model.addVar(vtype="B", name=f"Zone_{z}_Project_{p}") for z in range(number_of_zones) for p in range(number_of_projects)}
        maintenance_vars = {(z, p): model.addVar(vtype="C", lb=0, ub=maintenance_costs[z], name=f"Maintenance_Z_{z}_P_{p}") for z in range(number_of_zones) for p in range(number_of_projects)}

        # Objective: minimize the expected total cost including zone costs, project assignment costs, and maintenance costs
        model.setObjective(
            quicksum(zone_costs[z] * zone_vars[z] for z in range(number_of_zones)) +
            quicksum(project_costs[z, p] * project_vars[z, p] for z in range(number_of_zones) for p in range(number_of_projects)) +
            quicksum(maintenance_vars[z, p] * impact_values[z, p] for z in range(number_of_zones) for p in range(number_of_projects)) +
            (1 / no_of_scenarios) * quicksum(quicksum(planning_scenarios[s]['requirement'][p] * project_vars[z, p] for p in range(number_of_projects)) for z in range(number_of_zones) for s in range(no_of_scenarios)), "minimize"
        )
        
        # Constraints: Each project must be assigned to exactly one zone
        for p in range(number_of_projects):
            model.addCons(quicksum(project_vars[z, p] for z in range(number_of_zones)) == 1, f"Project_{p}_Requirement")
        
        # Constraints: Only selected zones can host projects
        for z in range(number_of_zones):
            for p in range(number_of_projects):
                model.addCons(project_vars[z, p] <= zone_vars[z], f"Zone_{z}_Host_{p}")

        # Constraints: Zones cannot exceed their project capacity in each scenario
        for s in range(no_of_scenarios):
            for z in range(number_of_zones):
                model.addCons(quicksum(planning_scenarios[s]['requirement'][p] * project_vars[z, p] for p in range(number_of_projects)) <= maintenance_costs[z] * zone_vars[z], f"Zone_{z}_Scenario_{s}_Capacity")

        # Constraints: Each zone must achieve a minimum impact value
        min_impact = 0.9
        for p in range(number_of_projects):
            model.addCons(quicksum(impact_values[z, p] * project_vars[z, p] for z in range(number_of_zones)) >= min_impact, f"Project_{p}_Impact")
        
        # Constraints: Each zone must not exceed the maximum number of projects
        max_projects = 5
        for z in range(number_of_zones):
            model.addCons(quicksum(project_vars[z, p] for p in range(number_of_projects)) <= zone_vars[z] * max_projects, f"Zone_{z}_MaxProjects")
 
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Zones': 150,
        'Number_of_Projects': 45,
        'Project_Cost_Lower_Bound': 225,
        'Project_Cost_Upper_Bound': 3000,
        'Min_Zone_Cost': 2529,
        'Max_Zone_Cost': 5000,
        'Min_Zone_Capacity': 39,
        'Max_Zone_Capacity': 472,
        'No_of_Scenarios': 20,
        'Requirement_Variation': 0.1,
        'Min_Impact': 0.52,
        'Max_Projects': 5,
    }

    urban_planning_optimizer = UrbanPlanning(parameters, seed=42)
    instance = urban_planning_optimizer.generate_instance()
    solve_status, solve_time, objective_value = urban_planning_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")