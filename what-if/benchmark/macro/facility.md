CODE PATH: facility.py

QUESTION: Why do we open the second plant?
CONSTRAINT CODE:
m.addConstr(open[1] == 0, 'Why the second plant')
TYPE: open-close

QUESTION: Why do we open plant {{VALUE-X}}? What if we close the plant  {{VALUE-X}}?
VALUE-X: random.randrange(len(capacity))
CONSTRAINT CODE:
m.addConstr(open[{{VALUE-X}}] == 0, 'Close {{VALUE-X}}')
TYPE: open-close


QUESTION: Why not open plant {{VALUE-X}}? Can we open plant {{VALUE-X}}? What if we open plant {{VALUE-X}}?
VALUE-X: random.randrange(len(capacity))
CONSTRAINT CODE:
m.addConstr(open[{{VALUE-X}}] == 1, 'Open {{VALUE-X}}')
TYPE: open-close


QUESTION: Why not select the edge from plant {{VALUE-X}} to warehouse {{VALUE-Y}}?
VALUE-X: random.randrange(len(capacity))
VALUE-Y: random.randrange(len(demand))
CONSTRAINT CODE:
m.addConstr(transport[{{VALUE-Y}}, {{VALUE-X}}] == 1, 'Why not select?')
TYPE: transport-yes-not


QUESTION: Why do we ship from plant {{VALUE-X}} to warehouse {{VALUE-Y}}?
VALUE-X: random.randrange(len(capacity))
VALUE-Y: random.randrange(len(demand))
CONSTRAINT CODE:
m.addConstr(transport[{{VALUE-Y}}, {{VALUE-X}}] == 0, 'Why select?')
TYPE: transport-yes-not

QUESTION: What if the shipping cost from plant {{VALUE-X}} to warehouse {{VALUE-Y}} is increased by {{VALUE-Z}}?
VALUE-X: random.randrange(len(capacity))
VALUE-Y: random.randrange(len(demand))
VALUE-Z: random.randrange(1, 2000)
DATA CODE:
transCosts[{{VALUE-Y}}][{{VALUE-X}}] += {{VALUE-Z}}
TYPE: transport-cost

QUESTION: What if the shipping cost from plant {{VALUE-X}} to warehouse {{VALUE-Y}} increased {{VALUE-Z}} times?
VALUE-X: random.randrange(len(capacity))
VALUE-Y: random.randrange(len(demand))
VALUE-Z: random.randrange(1, 4)
DATA CODE:
transCosts[{{VALUE-Y}}][{{VALUE-X}}] *= {{VALUE-Z}}
TYPE: transport-cost


QUESTION: What if the opening cost from plant {{VALUE-X}} is halved?
VALUE-X: random.randrange(len(capacity))
DATA CODE: fixedCosts[{{VALUE-X}}] *= 0.5
TYPE: open-cost

QUESTION: What if the demand is doubled?
DATA CODE:
demand = [v * 2 for v in demand]
TYPE: demand-value


QUESTION: What if the demand is increased by {{VALUE-X}}?
VALUE-X: random.randrange(1, 5)
DATA CODE:
demand = [v + {{VALUE-X}} for v in demand]
TYPE: demand-value
