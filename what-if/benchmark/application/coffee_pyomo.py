import time

from pyomo.environ import (ConcreteModel, Constraint, Integers, Objective,
                           SolverFactory, TerminationCondition, Var, minimize)

# Example data
capacity_in_supplier = {'supplier1': 150, 'supplier2': 50, 'supplier3': 100}
shipping_cost_from_supplier_to_roastery = {
    ('supplier1', 'roastery1'): 5,
    ('supplier1', 'roastery2'): 4,
    ('supplier2', 'roastery1'): 6,
    ('supplier2', 'roastery2'): 3,
    ('supplier3', 'roastery1'): 2,
    ('supplier3', 'roastery2'): 7
}
roasting_cost_light = {'roastery1': 3, 'roastery2': 5}
roasting_cost_dark = {'roastery1': 5, 'roastery2': 6}
shipping_cost_from_roastery_to_cafe = {
    ('roastery1', 'cafe1'): 5,
    ('roastery1', 'cafe2'): 3,
    ('roastery1', 'cafe3'): 6,
    ('roastery2', 'cafe1'): 4,
    ('roastery2', 'cafe2'): 5,
    ('roastery2', 'cafe3'): 2
}
light_coffee_needed_for_cafe = {'cafe1': 20, 'cafe2': 30, 'cafe3': 40}
dark_coffee_needed_for_cafe = {'cafe1': 20, 'cafe2': 20, 'cafe3': 100}

# Create a new model
model = ConcreteModel()

# OPTIGUIDE DATA CODE GOES HERE

# Variables
model.x = Var(shipping_cost_from_supplier_to_roastery.keys(),
              domain=Integers,
              bounds=(0, None))
model.y_light = Var(shipping_cost_from_roastery_to_cafe.keys(),
                    domain=Integers,
                    bounds=(0, None))
model.y_dark = Var(shipping_cost_from_roastery_to_cafe.keys(),
                   domain=Integers,
                   bounds=(0, None))

# Objective
model.obj = Objective(
    expr=sum(model.x[i] * shipping_cost_from_supplier_to_roastery[i]
             for i in shipping_cost_from_supplier_to_roastery) +
    sum(roasting_cost_light[r] * model.y_light[r, c] +
        roasting_cost_dark[r] * model.y_dark[r, c]
        for r, c in shipping_cost_from_roastery_to_cafe) + sum(
            (model.y_light[j] + model.y_dark[j]) *
            shipping_cost_from_roastery_to_cafe[j]
            for j in shipping_cost_from_roastery_to_cafe),
    sense=minimize)

# Constraints


def flow_constraint(model, r):
    return sum(model.x[i] for i in shipping_cost_from_supplier_to_roastery
               if i[1] == r) == sum(
                   model.y_light[j] + model.y_dark[j]
                   for j in shipping_cost_from_roastery_to_cafe if j[0] == r)


def supply_constraint(model, s):
    return sum(model.x[i] for i in shipping_cost_from_supplier_to_roastery
               if i[0] == s) <= capacity_in_supplier[s]


def light_demand_constraint(model, c):
    return sum(model.y_light[j] for j in shipping_cost_from_roastery_to_cafe
               if j[1] == c) >= light_coffee_needed_for_cafe[c]


def dark_demand_constraint(model, c):
    return sum(model.y_dark[j] for j in shipping_cost_from_roastery_to_cafe
               if j[1] == c) >= dark_coffee_needed_for_cafe[c]


roasteries = list(
    set(i[1] for i in shipping_cost_from_supplier_to_roastery.keys()))
suppliers = list(
    set(i[0] for i in shipping_cost_from_supplier_to_roastery.keys()))
cafes = list(set(i[1] for i in shipping_cost_from_roastery_to_cafe.keys()))

model.FlowConstraint = Constraint(roasteries, rule=flow_constraint)
model.SupplyConstraint = Constraint(suppliers, rule=supply_constraint)
model.LightDemandConstraint = Constraint(cafes, rule=light_demand_constraint)
model.DarkDemandConstraint = Constraint(cafes, rule=dark_demand_constraint)

solver = SolverFactory('glpk')
m = model

# OPTIGUIDE CONSTRAINT CODE GOES HERE

# Solve
# You can change the solver as per your requirement
m = solver.solve(model)

print(time.ctime())
if m.solver.termination_condition == TerminationCondition.optimal:
    print(f'Optimal cost: {model.obj()}')
else:
    print("Not solved to optimality. Optimization status:",
          m.solver.termination_condition)
