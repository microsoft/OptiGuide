import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EVChargingStationDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_neighborhoods_and_sites(self):
        neighborhoods = range(self.n_neighborhoods)
        sites = range(self.n_sites)

        # Deployment costs
        deployment_costs = np.random.randint(self.min_deployment_cost, self.max_deployment_cost + 1, (self.n_neighborhoods, self.n_sites))

        # Land space availability
        land_space = np.random.randint(self.min_land_space, self.max_land_space + 1, self.n_sites)

        # Neighborhood population & socioeconomic score
        neighborhood_population = np.random.randint(self.min_population, self.max_population + 1, self.n_neighborhoods)
        socioeconomic_scores = np.random.randint(self.min_socioecon_score, self.max_socioecon_score + 1, self.n_neighborhoods)

        # Distance between neighborhoods and sites
        distances = np.random.randint(self.min_distance, self.max_distance + 1, (self.n_neighborhoods, self.n_sites))

        res = {
            'neighborhoods': neighborhoods,
            'sites': sites,
            'deployment_costs': deployment_costs,
            'land_space': land_space,
            'neighborhood_population': neighborhood_population,
            'socioeconomic_scores': socioeconomic_scores,
            'distances': distances,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        neighborhoods = instance['neighborhoods']
        sites = instance['sites']
        deployment_costs = instance['deployment_costs']
        land_space = instance['land_space']
        neighborhood_population = instance['neighborhood_population']
        socioeconomic_scores = instance['socioeconomic_scores']
        distances = instance['distances']

        model = Model("EVChargingStationDeployment")
        
        # Variables
        deployment_vars = { (n, s): model.addVar(vtype="B", name=f"deployment_{n+1}_{s+1}") for n in neighborhoods for s in sites}
        site_vars = { s: model.addVar(vtype="B", name=f"site_{s+1}") for s in sites }
                
        # Objective
        # Minimize total deployment cost
        objective_expr = quicksum(deployment_costs[n, s] * deployment_vars[n, s] for n in neighborhoods for s in sites)
        
        model.setObjective(objective_expr, "minimize")
        
        ### Constraints
        # Ensure each neighborhood has access to at least one site
        for n in neighborhoods:
            model.addCons(quicksum(deployment_vars[n, s] for s in sites) >= 1, f"access_{n+1}")

        # Land space constraints for each site
        for s in sites:
            model.addCons(quicksum(deployment_vars[n, s] for n in neighborhoods) <= land_space[s], f"land_space_{s+1}")

        # Ensure equitable access based on socioeconomic scores
        for s in sites:
            model.addCons(quicksum(deployment_vars[n, s] * socioeconomic_scores[n] for n in neighborhoods) >= self.equity_threshold * socioeconomic_scores.mean(), f"socioecon_equity_{s+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 50,
        'n_sites': 300,
        'min_deployment_cost': 10000,
        'max_deployment_cost': 50000,
        'min_land_space': 1,
        'max_land_space': 60,
        'min_population': 2000,
        'max_population': 10000,
        'min_socioecon_score': 7,
        'max_socioecon_score': 400,
        'min_distance': 0,
        'max_distance': 400,
        'equity_threshold': 0.78,
    }

    ev_deployment = EVChargingStationDeployment(parameters, seed=seed)
    instance = ev_deployment.generate_neighborhoods_and_sites()
    solve_status, solve_time = ev_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")