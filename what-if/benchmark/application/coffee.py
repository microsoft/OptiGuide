import time
import string
from gurobipy import GRB, Model, quicksum
from coffee_instance import CoffeeInstance

# Create a new model
model = Model("coffee_distribution")

# Create instance
instance = CoffeeInstance()
instance.add_supplier('s1', 150)
instance.add_supplier('s2', 50)
instance.add_supplier('s3', 100)

instance.connect_supplier_to_roastery('s1', 'r1', 5)
instance.connect_supplier_to_roastery('s1', 'r2', 4)
instance.connect_supplier_to_roastery('s2', 'r1', 6)
instance.connect_supplier_to_roastery('s2', 'r2', 3)
instance.connect_supplier_to_roastery('s3', 'r1', 2)
instance.connect_supplier_to_roastery('s3', 'r2', 7)

instance.add_roastery('r1', 3, 5)
instance.add_roastery('r2', 5, 6)

instance.connect_roastery_to_cafe('r1', 'c1', 5)
instance.connect_roastery_to_cafe('r1', 'c2', 3)
instance.connect_roastery_to_cafe('r1', 'c3', 6)
instance.connect_roastery_to_cafe('r2', 'c1', 4)
instance.connect_roastery_to_cafe('r2', 'c2', 5)
instance.connect_roastery_to_cafe('r2', 'c3', 2)

instance.add_cafe('c1', 20, 20)
instance.add_cafe('c2', 30, 20)
instance.add_cafe('c3', 40, 100)

# OPTIGUIDE DATA CODE GOES HERE

# Create variables
x = model.addVars(instance.get_suppliers_to_roasteries(), vtype=GRB.INTEGER)
y_light = model.addVars(instance.get_roasteries_to_cafes(), vtype=GRB.INTEGER)
y_dark = model.addVars(instance.get_roasteries_to_cafes(), vtype=GRB.INTEGER)

# Set objective
model.setObjective(
    sum(cost * x[sr] for sr, cost in instance.shipping_cost_from_supplier_to_roastery.items()) +
    sum(cost * quicksum(y_light.select(r)) for r, cost in instance.roasting_cost_light.items()) + 
    sum(cost * quicksum(y_dark.select(r)) for r, cost in instance.roasting_cost_dark.items()) + 
    sum(cost * (y_light[rc] + y_dark[rc]) for rc, cost in instance.shipping_cost_from_roastery_to_cafe.items()), GRB.MINIMIZE)

# Conservation of flow constraint
for r in instance.roasteries:
    model.addConstr(quicksum(x.select('*', r)) == quicksum(y_light.select(r, '*')) + quicksum(y_dark.select(r, '*')))

# Add supply constraints
for s in instance.suppliers:
    model.addConstr(quicksum(x.select(s, '*')) <= instance.capacity_in_supplier[s])

# Add demand constraints
for c in instance.cafes:
    model.addConstr(quicksum(y_light.select('*', c)) >= instance.light_coffee_needed_for_cafe[c])
    model.addConstr(quicksum(y_dark.select('*', c)) >= instance.dark_coffee_needed_for_cafe[c])

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
        name = s
        capacity = instance.capacity_in_supplier[s]

        return f'''< 
<table border='0' cellborder='0' cellpadding="0" cellspacing="0">
    <tr>
        <td rowspan="2" port="link" width="40"><img src="app/static/supplier.png"/></td>
        <td align="left"><FONT FACE="sans-serif" POINT-SIZE="17"><B>{name}</B></FONT></td>
    </tr>
    <tr>
        <td align="left" valign="top"><FONT FACE="monospace" POINT-SIZE="8">{capacity}</FONT></td>
    </tr>
</table> >'''

    def roasterStr(r):
        name = r
        light = instance.roasting_cost_light[r]
        dark = instance.roasting_cost_dark[r]

        return f'''< 
<table border='0' cellborder='0' cellpadding="0" cellspacing="0">
    <tr>
        <td rowspan="3" port="link" width="50"><img src="app/static/roaster.png"/></td>
        <td align="left" VALIGN="TOP"><FONT FACE="sans-serif" POINT-SIZE="18"><B>{name}</B></FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="light"><FONT FACE="monospace" POINT-SIZE="8">{light}</FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="dark"><FONT FACE="monospace" POINT-SIZE="8"><B>{dark}</B></FONT></td>
    </tr>
</table> >'''

    def cafeStr(c):
        name = c
        light = instance.light_coffee_needed_for_cafe[c]
        dark = instance.dark_coffee_needed_for_cafe[c]

        return f'''< 
<table border='0' cellborder='0' cellpadding="0" cellspacing="0">
    <tr>
        <td rowspan="3" port="link" valign="middle"><img src="app/static/cafe.png"/></td>
        <td align="left"><FONT FACE="sans-serif" POINT-SIZE="18"><B>{name}</B></FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="light"><FONT FACE="monospace" POINT-SIZE="8">{int(light)}</FONT></td>
    </tr>
    <tr>
        <td ALIGN="LEFT" port="dark"><FONT FACE="monospace" POINT-SIZE="8"><B>{int(dark)}</B></FONT></td>
    </tr>
</table> >'''


    return f'''
digraph {{
    graph [layoutType="Sugiyama LR",rankdir=TB,splines=true];
    {{ rank=same {';'.join(instance.suppliers)}; }}
    {{ rank=same {';'.join(instance.roasteries)}; }}
    {{ rank=same {';'.join(instance.cafes)}; }}
    
    {" ".join([f'{s} [label={supplierStr(s)}, shape=none];' for s in instance.suppliers])}
    {" ".join([f'{r} [label={roasterStr(r)}, shape=none];' for r in instance.roasteries])}
    {" ".join([f'{c} [label={cafeStr(c)}, shape=none];' for c in instance.cafes])}

    {"; ".join([f'{i[0]} -> {i[1]} [label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0"><tr><td align="left"><FONT FACE="monospace" POINT-SIZE="8">{int(x[i].X)}</FONT></td></tr></table>> {"" if x[i].X > 0 else ",style=invis"}]' for i,d in instance.shipping_cost_from_supplier_to_roastery.items()])};
    {"; ".join([f'{i[0]} -> {i[1]} [label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0"><tr><td align="left"><FONT FACE="monospace" POINT-SIZE="8">{int(y_light[i].X)}<BR/><B>{int(y_dark[i].X)}</B></FONT></td></tr></table>> {"" if (y_light[i].X > 0 or y_dark[i].X > 0) else ",style=invis"}]' for i,d in instance.shipping_cost_from_roastery_to_cafe.items()])};

    edge [style=invis]
    {" -> ".join([i for i in sorted(instance.suppliers)])};
    {" -> ".join([i for i in sorted(instance.roasteries)])};
    {" -> ".join([i for i in sorted(instance.cafes)])};
}}'''


instance_graph = generate_graph()