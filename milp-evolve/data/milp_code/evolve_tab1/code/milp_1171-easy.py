import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnvironmentalMonitoringSensorPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def get_instance(self):
        assert self.nSensors > 0 and self.nMonitoringNodes > self.nSensors
        assert self.maxCommunicationRange > 0 and self.budgetLimit >= 0

        sensorPlacementCost = np.random.randint(8000, 30000, self.nSensors)
        operationalCost = np.random.uniform(300, 1200, self.nSensors)
        monitoringNodes = np.random.randint(10, 300, self.nMonitoringNodes)
        
        communicationDistances = np.abs(np.random.normal(loc=300, scale=100, size=(self.nSensors, self.nMonitoringNodes)))
        communicationDistances = np.where(communicationDistances > self.maxCommunicationRange, self.maxCommunicationRange, communicationDistances)
        communicationDistances = np.where(communicationDistances < 1, 1, communicationDistances)

        network_graph = self.generate_random_graph()
        
        return {
            "sensorPlacementCost": sensorPlacementCost,
            "operationalCost": operationalCost,
            "monitoringNodes": monitoringNodes,
            "communicationDistances": communicationDistances,
            "graph": network_graph,
        }
    
    def generate_random_graph(self):
        n_nodes = self.nMonitoringNodes
        G = nx.barabasi_albert_graph(n=n_nodes, m=3, seed=self.seed)
        return G

    def solve(self, instance):
        sensorPlacementCost = instance['sensorPlacementCost']
        operationalCost = instance['operationalCost']
        monitoringNodes = instance['monitoringNodes']
        communicationDistances = instance['communicationDistances']
        G = instance['graph']

        model = Model("EnvironmentalMonitoringSensorPlacement")

        totalSensors = len(sensorPlacementCost)
        totalNodes = len(monitoringNodes)

        sensor_vars = {s: model.addVar(vtype="B", name=f"Sensor_{s}") for s in range(totalSensors)}
        areaCoverage_vars = {(s, n): model.addVar(vtype="B", name=f"AreaCoverage_{s}_{n}") for s in range(totalSensors) for n in range(totalNodes)}

        for s in range(totalSensors):
            model.addCons(
                quicksum(communicationDistances[s, n] * areaCoverage_vars[s, n] for n in range(totalNodes)) <= self.maxCommunicationRange,
                f"CommunicationLimit_{s}"
            )

        for n in range(totalNodes):
            model.addCons(
                quicksum(areaCoverage_vars[s, n] for s in range(totalSensors)) >= 1,
                f"NodeCoverage_{n}"
            )

        for s in range(totalSensors):
            for n in range(totalNodes):
                model.addCons(
                    areaCoverage_vars[s, n] <= sensor_vars[s],
                    f"Coverage_{s}_{n}"
                )

        # Symmetry Breaking Constraints
        for s in range(totalSensors - 1):
            model.addCons(
                sensor_vars[s] >= sensor_vars[s + 1],
                f"Symmetry_{s}"
            )
        
        # Hierarchical Demand Fulfillment
        for n in range(totalNodes):
            for s1 in range(totalSensors):
                for s2 in range(s1 + 1, totalSensors):
                    model.addCons(
                        areaCoverage_vars[s2, n] <= areaCoverage_vars[s1, n] + int(communicationDistances[s1, n] <= self.maxCommunicationRange),
                        f"Hierarchical_{s1}_{s2}_{n}"
                    )

        ### New Constraints: Frequency of Data Collection
        for s in range(totalSensors):
            model.addCons(
                quicksum(areaCoverage_vars[s, n] * monitoringNodes[n] for n in range(totalNodes)) >= self.dataFrequency,
                f"DataFrequency_{s}"
            )
        
        model.setObjective(
            quicksum(sensorPlacementCost[s] * sensor_vars[s] for s in range(totalSensors)) + 
            quicksum(communicationDistances[s, n] * areaCoverage_vars[s, n] for s in range(totalSensors) for n in range(totalNodes)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'nSensors': 60,
        'nMonitoringNodes': 180,
        'budgetLimit': 100000,
        'maxCommunicationRange': 1200,
        'dataFrequency': 1000,
    }

    monitoring_sensor_solver = EnvironmentalMonitoringSensorPlacement(parameters, seed=42)
    instance = monitoring_sensor_solver.get_instance()
    solve_status, solve_time, objective_value = monitoring_sensor_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")