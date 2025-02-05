import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WasteManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.Number_of_Trucks > 0 and self.Number_of_Areas > 0
        assert self.Min_Truck_Cost >= 0 and self.Max_Truck_Cost >= self.Min_Truck_Cost
        assert self.Min_Emissions >= 0 and self.Max_Emissions >= self.Min_Emissions
        
        truck_costs = np.random.randint(self.Min_Truck_Cost, self.Max_Truck_Cost + 1, self.Number_of_Trucks)
        emissions = np.random.uniform(self.Min_Emissions, self.Max_Emissions, self.Number_of_Trucks)
        area_generation_rates = np.random.randint(self.Min_Waste_Generation, self.Max_Waste_Generation + 1, self.Number_of_Areas)
        area_recycling_rates = np.random.randint(self.Min_Recycling_Rate, self.Max_Recycling_Rate + 1, self.Number_of_Areas)
        disposal_site_capacities = np.random.randint(self.Min_Disposal_Capacity, self.Max_Disposal_Capacity + 1, self.Number_of_Disposal_Sites)
        transportation_costs = np.random.randint(self.Min_Transportation_Cost, self.Max_Transportation_Cost + 1, (self.Number_of_Trucks, self.Number_of_Areas))
        
        return {
            "truck_costs": truck_costs,
            "emissions": emissions,
            "area_generation_rates": area_generation_rates,
            "area_recycling_rates": area_recycling_rates,
            "disposal_site_capacities": disposal_site_capacities,
            "transportation_costs": transportation_costs,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        truck_costs = instance['truck_costs']
        emissions = instance['emissions']
        area_generation_rates = instance['area_generation_rates']
        area_recycling_rates = instance['area_recycling_rates']
        disposal_site_capacities = instance['disposal_site_capacities']
        transportation_costs = instance['transportation_costs']
        
        model = Model("WasteManagement")
        number_of_trucks = len(truck_costs)
        number_of_areas = len(area_generation_rates)
        number_of_disposal_sites = len(disposal_site_capacities)

        # Decision variables
        truck_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(number_of_trucks)}
        route_vars = {(t, a): model.addVar(vtype="B", name=f"Truck_{t}_Area_{a}") for t in range(number_of_trucks) for a in range(number_of_areas)}
        recycling_vars = {a: model.addVar(vtype="C", name=f"Recycling_Area_{a}") for a in range(number_of_areas)}
        disposal_vars = {(a, s): model.addVar(vtype="C", name=f"Area_{a}_Disposal_{s}") for a in range(number_of_areas) for s in range(number_of_disposal_sites)}

        # Objective: minimize the total cost including truck usage costs, transportation costs, and emission costs
        model.setObjective(
            quicksum(truck_costs[t] * truck_vars[t] for t in range(number_of_trucks)) +
            quicksum(transportation_costs[t, a] * route_vars[t, a] for t in range(number_of_trucks) for a in range(number_of_areas)) +
            quicksum(emissions[t] * truck_vars[t] for t in range(number_of_trucks)), "minimize"
        )
        
        # Constraints: Each area waste generation should be either recycled or disposed
        for a in range(number_of_areas):
            model.addCons(
                quicksum(recycling_vars[a] + disposal_vars[(a, s)] for s in range(number_of_disposal_sites)) == area_generation_rates[a], f"Area_Waste_Gen_{a}"
            )
        
        # Constraints: Recycling should not exceed the generation rates
        for a in range(number_of_areas):
            model.addCons(recycling_vars[a] <= area_recycling_rates[a], f"Area_Recycling_{a}")
        
        # Constraints: Each disposal site must not exceed its capacity
        for s in range(number_of_disposal_sites):
            model.addCons(
                quicksum(disposal_vars[(a, s)] for a in range(number_of_areas)) <= disposal_site_capacities[s], f"Disposal_Capacity_{s}"
            )
        
        # Set covering constraint for each area
        for a in range(number_of_areas):
            model.addCons(quicksum(route_vars[t, a] for t in range(number_of_trucks)) >= 1, f"Set_Covering_Area_{a}")
        
        # Constraints: Only active trucks can serve areas
        for t in range(number_of_trucks):
            for a in range(number_of_areas):
                model.addCons(route_vars[t, a] <= truck_vars[t], f"Truck_{t}_Serve_{a}")

        # Solve the model
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Trucks': 450,
        'Number_of_Areas': 37,
        'Number_of_Disposal_Sites': 90,
        'Min_Waste_Generation': 500,
        'Max_Waste_Generation': 1500,
        'Min_Recycling_Rate': 250,
        'Max_Recycling_Rate': 2025,
        'Min_Disposal_Capacity': 3000,
        'Max_Disposal_Capacity': 5000,
        'Min_Transportation_Cost': 900,
        'Max_Transportation_Cost': 3000,
        'Min_Truck_Cost': 5000,
        'Max_Truck_Cost': 10000,
        'Min_Emissions': 75,
        'Max_Emissions': 350,
    }

    waste_management_optimizer = WasteManagement(parameters, seed)
    instance = waste_management_optimizer.generate_instance()
    solve_status, solve_time, objective_value = waste_management_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")