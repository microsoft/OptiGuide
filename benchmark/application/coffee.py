import time
from gurobipy import GRB, Model

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

cafes = list(set(i[1] for i in shipping_cost_from_roastery_to_cafe.keys()))
roasteries = list(
    set(i[1] for i in shipping_cost_from_supplier_to_roastery.keys()))
suppliers = list(
    set(i[0] for i in shipping_cost_from_supplier_to_roastery.keys()))

# Create a new model
model = Model("coffee_distribution")

# OPTIGUIDE DATA CODE GOES HERE

# Create variables
x = model.addVars(shipping_cost_from_supplier_to_roastery.keys(),
                  vtype=GRB.INTEGER,
                  name="x")
y_light = model.addVars(shipping_cost_from_roastery_to_cafe.keys(),
                        vtype=GRB.INTEGER,
                        name="y_light")
y_dark = model.addVars(shipping_cost_from_roastery_to_cafe.keys(),
                       vtype=GRB.INTEGER,
                       name="y_dark")

# Set objective
model.setObjective(
    sum(x[i] * shipping_cost_from_supplier_to_roastery[i]
        for i in shipping_cost_from_supplier_to_roastery.keys()) +
    sum(roasting_cost_light[r] * y_light[r, c] +
        roasting_cost_dark[r] * y_dark[r, c]
        for r, c in shipping_cost_from_roastery_to_cafe.keys()) + sum(
            (y_light[j] + y_dark[j]) * shipping_cost_from_roastery_to_cafe[j]
            for j in shipping_cost_from_roastery_to_cafe.keys()), GRB.MINIMIZE)

# Conservation of flow constraint
for r in set(i[1] for i in shipping_cost_from_supplier_to_roastery.keys()):
    model.addConstr(
        sum(x[i] for i in shipping_cost_from_supplier_to_roastery.keys()
            if i[1] == r) == sum(
                y_light[j] + y_dark[j]
                for j in shipping_cost_from_roastery_to_cafe.keys()
                if j[0] == r), f"flow_{r}")

# Add supply constraints
for s in set(i[0] for i in shipping_cost_from_supplier_to_roastery.keys()):
    model.addConstr(
        sum(x[i] for i in shipping_cost_from_supplier_to_roastery.keys()
            if i[0] == s) <= capacity_in_supplier[s], f"supply_{s}")

# Add demand constraints
for c in set(i[1] for i in shipping_cost_from_roastery_to_cafe.keys()):
    model.addConstr(
        sum(y_light[j] for j in shipping_cost_from_roastery_to_cafe.keys()
            if j[1] == c) >= light_coffee_needed_for_cafe[c],
        f"light_demand_{c}")
    model.addConstr(
        sum(y_dark[j] for j in shipping_cost_from_roastery_to_cafe.keys()
            if j[1] == c) >= dark_coffee_needed_for_cafe[c],
        f"dark_demand_{c}")

# Optimize model
model.optimize()
m = model

# OPTIGUIDE CONSTRAINT CODE GOES HERE

# Solve
m.update()
model.optimize()

print(time.ctime())
if m.status == GRB.OPTIMAL:
    print(f'Optimal cost: {m.objVal}')
else:
    print("Not solved to optimality. Optimization status:", m.status)
