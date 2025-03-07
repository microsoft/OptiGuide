import time
import string
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
roasteries = list(set(i[1] for i in shipping_cost_from_supplier_to_roastery.keys()))
suppliers = list(set(i[0] for i in shipping_cost_from_supplier_to_roastery.keys()))

# Create a new model
model = Model("coffee_distribution")

# OPTIGUIDE DATA CODE GOES HERE

# Create variables
x = model.addVars(shipping_cost_from_supplier_to_roastery.keys(), vtype=GRB.INTEGER, name="x")
y_light = model.addVars(shipping_cost_from_roastery_to_cafe.keys(), vtype=GRB.INTEGER, name="y_light")
y_dark = model.addVars(shipping_cost_from_roastery_to_cafe.keys(), vtype=GRB.INTEGER, name="y_dark")

# Set objective
model.setObjective(
    sum(x[i] * shipping_cost_from_supplier_to_roastery[i] for i in shipping_cost_from_supplier_to_roastery.keys()) +
    sum(roasting_cost_light[r] * y_light[r, c] + roasting_cost_dark[r] * y_dark[r, c]
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


##### STOP HERE
# Generate visualization code
def generate_graph():
    def supplierStr(s):
        name = int(s.strip(string.ascii_letters))
        capacity = capacity_in_supplier[s]

        return f'''< 
<table border='0' cellborder='0' cellpadding="0" cellspacing="0">
    <tr>
        <td rowspan="2" port="link" width="40"><img src="app/static/supplier.png"/></td>
        <td align="left"><FONT FACE="sans-serif" POINT-SIZE="17"><B>s{name}</B></FONT></td>
    </tr>
    <tr>
        <td align="left" valign="top"><FONT FACE="monospace" POINT-SIZE="8">{capacity}</FONT></td>
    </tr>
</table> >'''

    def roasterStr(r):
        name = int(r.strip(string.ascii_letters))
        light = roasting_cost_light[r]
        dark = roasting_cost_dark[r]

        return f'''< 
<table border='0' cellborder='0' cellpadding="0" cellspacing="0">
    <tr>
        <td rowspan="3" port="link" width="50"><img src="app/static/roaster.png"/></td>
        <td align="left" VALIGN="TOP"><FONT FACE="sans-serif" POINT-SIZE="18"><B>r{name}</B></FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="light"><FONT FACE="monospace" POINT-SIZE="8">{light}</FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="dark"><FONT FACE="monospace" POINT-SIZE="8"><B>{dark}</B></FONT></td>
    </tr>
</table> >'''

    def cafeStr(c):
        name = int(c.strip(string.ascii_letters))
        light = light_coffee_needed_for_cafe[c]
        dark = dark_coffee_needed_for_cafe[c]

        return f'''< 
<table border='0' cellborder='0' cellpadding="0" cellspacing="0">
    <tr>
        <td rowspan="3" port="link" valign="middle"><img src="app/static/cafe.png"/></td>
        <td align="left"><FONT FACE="sans-serif" POINT-SIZE="18"><B>c{name}</B></FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="light"><FONT FACE="monospace" POINT-SIZE="8">{light}</FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="dark"><FONT FACE="monospace" POINT-SIZE="8"><B>{dark}</B></FONT></td>
    </tr>
</table> >'''


    return f'''
digraph {{
    graph [layoutType="Sugiyama LR",rankdir=TB,splines=true];
    {{ rank=same {';'.join(suppliers)}; }}
    {{ rank=same {';'.join(roasteries)}; }}
    {{ rank=same {';'.join(cafes)}; }}
    
    {" ".join([f'{s} [label={supplierStr(s)}, shape=none];' for s in suppliers])}
    {" ".join([f'{r} [label={roasterStr(r)}, shape=none];' for r in roasteries])}
    {" ".join([f'{c} [label={cafeStr(c)}, shape=none];' for c in cafes])}

    {"; ".join([f'{i[0]} -> {i[1]} [label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0"><tr><td align="left"><FONT FACE="monospace" POINT-SIZE="8">{int(x[i].X)}</FONT></td></tr></table>> {"" if x[i].X > 0 else ",style=invis"}]' for i,d in shipping_cost_from_supplier_to_roastery.items()])};
    {"; ".join([f'{i[0]} -> {i[1]} [label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0"><tr><td align="left"><FONT FACE="monospace" POINT-SIZE="8">{int(y_light[i].X)}<BR/><B>{int(y_dark[i].X)}</B></FONT></td></tr></table>> {"" if (y_light[i].X > 0 or y_dark[i].X > 0) else ",style=invis"}]' for i,d in shipping_cost_from_roastery_to_cafe.items()])};

    edge [style=invis]
    {" -> ".join([i for i in sorted(suppliers)])};
    {" -> ".join([i for i in sorted(roasteries)])};
    {" -> ".join([i for i in sorted(cafes)])};
}}'''


instance_graph = generate_graph()