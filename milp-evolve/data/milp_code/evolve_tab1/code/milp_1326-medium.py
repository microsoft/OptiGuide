import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EMSNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        assert self.nStations > 0 and self.nZones > 0
        assert self.maximumVehicles > 0 and self.hazardousMaterialCost >= 0

        stationOperatingCosts = np.random.randint(15000, 60000, self.nStations)
        zoneDemands = np.random.randint(10, 100, self.nZones)
        responseTimes = np.abs(np.random.normal(loc=100, scale=20, size=(self.nStations, self.nZones)))
        responseTimes = np.where(responseTimes < 1, 1, responseTimes)
        hazardousMaterialPenalty = np.random.randint(100, 1000, self.nZones)

        renewable_energy_costs = np.random.randint(50, 500, (self.nStations, self.nZones))
        carbon_emissions = np.random.randint(1, 10, (self.nStations, self.nZones))

        # Piecewise linear segments data
        segments = 3  # Number of segments for piecewise linear functions
        piecewise_points = np.linspace(0, 1, segments + 1)  # Example points: [0, 0.33, 0.66, 1]
        piecewise_slopes = np.random.rand(segments) * 10  # Random slopes for each segment
        piecewise_slopes_energy = np.random.rand(segments) * 5
        piecewise_slopes_emissions = np.random.rand(segments) * 2

        return {
            "stationOperatingCosts": stationOperatingCosts,
            "zoneDemands": zoneDemands,
            "responseTimes": responseTimes,
            "hazardousMaterialPenalty": hazardousMaterialPenalty,
            "renewable_energy_costs": renewable_energy_costs,
            "carbon_emissions": carbon_emissions,
            "piecewise_points": piecewise_points,
            "piecewise_slopes": piecewise_slopes,
            "piecewise_slopes_energy": piecewise_slopes_energy,
            "piecewise_slopes_emissions": piecewise_slopes_emissions,
            "segments": segments
        }

    def solve(self, instance):
        stationOperatingCosts = instance['stationOperatingCosts']
        zoneDemands = instance['zoneDemands']
        responseTimes = instance['responseTimes']
        hazardousMaterialPenalty = instance['hazardousMaterialPenalty']
        renewable_energy_costs = instance['renewable_energy_costs']
        carbon_emissions = instance['carbon_emissions']
        piecewise_points = instance["piecewise_points"]
        piecewise_slopes = instance["piecewise_slopes"]
        piecewise_slopes_energy = instance["piecewise_slopes_energy"]
        piecewise_slopes_emissions = instance["piecewise_slopes_emissions"]
        segments = instance["segments"]

        model = Model("EMSNetworkOptimization")

        nStations = len(stationOperatingCosts)
        nZones = len(zoneDemands)

        station_vars = {s: model.addVar(vtype="B", name=f"Station_{s}") for s in range(nStations)}
        zone_assignments = {(z, s): model.addVar(vtype="B", name=f"Assignment_{z}_{s}") for z in range(nZones) for s in range(nStations)}
        renewable_energy_vars = {(z, s): model.addVar(vtype="B", name=f"Renewable_{z}_{s}") for z in range(nZones) for s in range(nStations)}

        # Piecewise linear variables
        response_piecewise = {(z, s, k): model.addVar(vtype="C", name=f"ResponsePiecewise_{z}_{s}_{k}") for z in range(nZones) for s in range(nStations) for k in range(segments)}
        energy_piecewise = {(z, s, k): model.addVar(vtype="C", name=f"EnergyPiecewise_{z}_{s}_{k}") for z in range(nZones) for s in range(nStations) for k in range(segments)}
        emissions_piecewise = {(z, s, k): model.addVar(vtype="C", name=f"EmissionsPiecewise_{z}_{s}_{k}") for z in range(nZones) for s in range(nStations) for k in range(segments)}

        for z in range(nZones):
            model.addCons(
                quicksum(zone_assignments[z, s] for s in range(nStations)) == 1,
                f"ZoneAssignment_{z}"
            )

        for s in range(nStations):
            model.addCons(
                quicksum(zoneDemands[z] * zone_assignments[z, s] for z in range(nZones)) <= self.maximumVehicles * station_vars[s],
                f"StationCapacity_{s}"
            )

        for z in range(nZones):
            for s in range(nStations):
                model.addCons(
                    zone_assignments[z, s] <= station_vars[s],
                    f"AssignmentLimit_{z}_{s}"
                )

        for s in range(nStations - 1):
            model.addCons(
                station_vars[s] >= station_vars[s + 1],
                f"Symmetry_{s}"
            )

        for z in range(nZones):
            model.addCons(
                quicksum(responseTimes[s, z] * zone_assignments[z, s] for s in range(nStations)) <= self.hazardousMaterialCost + hazardousMaterialPenalty[z],
                f"HazardousMaterial_{z}"
            )

        for z in range(nZones):
            for s in range(nStations):
                model.addCons(
                    renewable_energy_vars[z, s] <= zone_assignments[z, s],
                    f"RenewableEnergyLimit_{z}_{s}"
                )

        model.addCons(
            quicksum(renewable_energy_vars[z, s] for z in range(nZones) for s in range(nStations)) >= self.renewable_percentage * quicksum(zone_assignments[z, s] for z in range(nZones) for s in range(nStations)),
            "RenewablePercentage"
        )

        model.addCons(
            quicksum(carbon_emissions[s, z] * zone_assignments[z, s] for z in range(nZones) for s in range(nStations)) <= self.carbon_limit,
            "CarbonLimit"
        )

        # Piecewise linear constraints
        for z in range(nZones):
            for s in range(nStations):
                for k in range(segments):
                    # Response times
                    model.addCons(
                        response_piecewise[z, s, k] <= piecewise_slopes[k] * zone_assignments[z, s],
                        f"ResponseSegment_{z}_{s}_{k}"
                    )
                    # Renewable energy costs
                    model.addCons(
                        energy_piecewise[z, s, k] <= piecewise_slopes_energy[k] * renewable_energy_vars[z, s],
                        f"EnergySegment_{z}_{s}_{k}"
                    )
                    # Carbon emissions
                    model.addCons(
                        emissions_piecewise[z, s, k] <= piecewise_slopes_emissions[k] * zone_assignments[z, s],
                        f"EmissionsSegment_{z}_{s}_{k}"
                    )

        model.setObjective(
            quicksum(stationOperatingCosts[s] * station_vars[s] for s in range(nStations)) +
            quicksum(responseTimes[s, z] * zone_assignments[z, s] for z in range(nZones) for s in range(nStations)) +
            quicksum(renewable_energy_costs[s, z] * renewable_energy_vars[z, s] for z in range(nZones) for s in range(nStations)) +
            quicksum(response_piecewise[z, s, k] for z in range(nZones) for s in range(nStations) for k in range(segments)) +
            quicksum(energy_piecewise[z, s, k] for z in range(nZones) for s in range(nStations) for k in range(segments)) +
            quicksum(emissions_piecewise[z, s, k] for z in range(nZones) for s in range(nStations) for k in range(segments)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    parameters = {
        'nStations': 60,
        'nZones': 75,
        'maximumVehicles': 1350,
        'hazardousMaterialCost': 75,
        'complianceRate': 0.8,
        'renewable_percentage': 0.24,
        'carbon_limit': 3000,
    }

    seed = 42

    ems_solver = EMSNetworkOptimization(parameters, seed=seed)
    instance = ems_solver.get_instance()
    solve_status, solve_time, objective_value = ems_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")