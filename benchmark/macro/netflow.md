CODE PATH: netflow.py


QUESTION: What if the demand of {{VALUE-P}} at {{VALUE-X}} is reduced by {{VALUE-COUNT}}?
VALUE-P: random.choice(commodities)
VALUE-X: random.choice(nodes)
VALUE-COUNT: random.randrange(1, 10)
DATA CODE:
inflow[{{VALUE-P}}, {{VALUE-X}}] += {{VALUE-COUNT}} if ({{VALUE-P}}, {{VALUE-X}}) in inflow else {{VALUE-COUNT}}
TYPE: demand-supply-data


QUESTION: What if the supply of pencil at {{VALUE-X}} is increased by {{VALUE-P}}%?
VALUE-X: random.choice(nodes)
VALUE-P: random.randrange(0, 50)
DATA CODE:
if ({{VALUE-P}}, {{VALUE-X}}) in inflow:
    inflow[{{VALUE-P}}, {{VALUE-X}}] *= (1 + {{VALUE-P}}/100)
TYPE: demand-supply-data


QUESTION: What if we reduce the shipping cost from {{VALUE-X}} by half?
VALUE-X: random.choice(nodes)
DATA CODE:
for a in commodities:
    for b in nodes:
        if (a, {{VALUE-X}}, b) in cost:
                cost[a, {{VALUE-X}}, b] *= 0.5
TYPE: shipping-cost


QUESTION: What if the shipping cost to {{VALUE-X}} would be increased by {{VALUE-C}}?
VALUE-X: random.choice(nodes)
VALUE-C: random.randrange(0, 50)
DATA CODE:
for a in commodities:
    for b in nodes:
        if (a, b, {{VALUE-X}}) in cost:
                cost[a, b, {{VALUE-X}}] += {{VALUE-C}}
TYPE: shipping-cost


QUESTION: Why would we ship {{VALUE-P}} from {{VALUE-X}} to {{VALUE-Y}}?
VALUE-IDX: random.randint(0, len(original_solution) - 1)
VALUE-P: original_solution[{{VALUE-IDX}}][0]
VALUE-X: original_solution[{{VALUE-IDX}}][1]
VALUE-Y: original_solution[{{VALUE-IDX}}][2]
CONSTRAINT CODE:
if ({{VALUE-P}}, {{VALUE-X}}, {{VALUE-Y}}) in flow:
    m.addConstr(flow[{{VALUE-P}}, {{VALUE-X}}, {{VALUE-Y}}] == 0, 'Why {{VALUE-P}} {{VALUE-X}} to {{VALUE-Y}}?')
TYPE: shipment


QUESTION: Why not ship {{VALUE-P}} from {{VALUE-X}} to {{VALUE-Y}}?
VALUE-P: random.choice(commodities)
VALUE-X: random.choice([src for product, src, dest in cost.keys() if product == {{VALUE-P}}])
VALUE-Y: random.choice([dest for product, src, dest in cost.keys() if product == {{VALUE-P}}])
CONSTRAINT CODE:
if ({{VALUE-P}}, {{VALUE-X}}, {{VALUE-Y}}) in flow:
    m.addConstr(flow[{{VALUE-P}}, {{VALUE-X}}, {{VALUE-Y}}] == 1, 'Why {{VALUE-P}} {{VALUE-X}} to {{VALUE-Y}}?')
TYPE: shipment

QUESTION: What if the capacity from {{VALUE-X}} to {{VALUE-Y}} doubled?
VALUE-X: random.choice(nodes)
VALUE-Y: random.choice(list(set(nodes) - set([{{VALUE-X}}])))
DATA CODE:
if ({{VALUE-X}}, {{VALUE-Y}}) in capacity:
    capacity[{{VALUE-X}}, {{VALUE-Y}}] *= 2
TYPE: shipment-capacity


QUESTION: What if we can ship {{VALUE-P}} from {{VALUE-X}} to {{VALUE-Y}} with a cost of {{VALUE-COST}}?
VALUE-P: random.choice(commodities)
VALUE-X: random.choice(nodes)
VALUE-Y: random.choice(list(set(nodes) - set([{{VALUE-X}}])))
VALUE-COST: random.randrange(1, 50)
DATA CODE:
cost[{{VALUE-P}}, {{VALUE-X}}, {{VALUE-Y}}] = {{VALUE-COST}}
TYPE: shipment-cost
