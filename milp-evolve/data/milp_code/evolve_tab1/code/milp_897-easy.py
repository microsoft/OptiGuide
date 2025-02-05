import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ConferenceRoomScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.min_capacity >= 0 and self.max_capacity >= self.min_capacity

        # Generate room capacities
        room_capacities = self.min_capacity + (self.max_capacity - self.min_capacity) * np.random.rand(self.number_of_rooms)

        meetings = []

        # Create meeting schedules
        for _ in range(self.number_of_meetings):
            required_capacity = self.min_capacity + (self.max_capacity - self.min_capacity) * np.random.rand()
            start_time = random.randint(0, self.max_time - self.meeting_duration)
            end_time = start_time + self.meeting_duration

            meetings.append((required_capacity, start_time, end_time))

        room_availability = [[] for room in range(self.number_of_rooms)]
        for i, (required_capacity, start_time, end_time) in enumerate(meetings):
            for room in range(self.number_of_rooms):
                if room_capacities[room] >= required_capacity:
                    room_availability[room].append(i)

        return {
            "meetings": meetings,
            "room_availability": room_availability,
            "room_capacities": room_capacities
        }
    
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        meetings = instance['meetings']
        room_availability = instance['room_availability']

        model = Model("ConferenceRoomScheduling")

        # Decision variables
        schedule_vars = {(r, i): model.addVar(vtype="B", name=f"Room_{r}_Meeting_{i}") for r in range(self.number_of_rooms) for i in range(len(meetings))}

        # Objective: maximize the number of meetings scheduled
        objective_expr = quicksum(schedule_vars[r, i] for r in range(self.number_of_rooms) for i in range(len(meetings)) if i in room_availability[r])

        # Constraints: Each room can host only one meeting at a given time
        for r in range(self.number_of_rooms):
            for i1 in range(len(meetings)):
                if i1 not in room_availability[r]:
                    continue
                for i2 in range(i1 + 1, len(meetings)):
                    if i2 not in room_availability[r]:
                        continue
                    if meetings[i1][1] < meetings[i2][2] and meetings[i2][1] < meetings[i1][2]:
                        model.addCons(schedule_vars[r, i1] + schedule_vars[r, i2] <= 1, f"Room_{r}_Conflict_{i1}_{i2}")

        # Constraints: Each meeting should be scheduled in at most one room
        for i in range(len(meetings)):
            model.addCons(quicksum(schedule_vars[r, i] for r in range(self.number_of_rooms) if i in room_availability[r]) <= 1, f"Meeting_{i}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_rooms': 30,
        'number_of_meetings': 125,
        'min_capacity': 30,
        'max_capacity': 300,
        'meeting_duration': 20,
        'max_time': 72,
    }

    scheduler = ConferenceRoomScheduling(parameters, seed)
    instance = scheduler.generate_instance()
    solve_status, solve_time = scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")