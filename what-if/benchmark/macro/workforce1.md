CODE PATH: workforce1.py

QUESTION:
What if {{VALUE-NAME}} does not work on {{VALUE-DAY}}?
VALUE-NAME: random.choice(workers)
VALUE-DAY: random.choice(shifts)
CONSTRAINT CODE:
if ({{VALUE-NAME}}, {{VALUE-DAY}}) in x:
    m.addConstr(x[{{VALUE-NAME}}, {{VALUE-DAY}}] == 0, "_")
TYPE: constraint-availability

QUESTION:
Can {{VALUE-NAME}} take {{VALUE-DAY1}} off and work on {{VALUE-DAY2}}?
VALUE-NAME: random.choice(workers)
VALUE-DAY1: random.choice(shifts)
VALUE-DAY2: random.choice(list(set(shifts) - set([{{VALUE-DAY1}}])))
CONSTRAINT CODE:
if ({{VALUE-NAME}}, {{VALUE-DAY1}}) in x:
    m.addConstr(x[{{VALUE-NAME}}, {{VALUE-DAY1}}] == 0, "_")
if ({{VALUE-NAME}}, {{VALUE-DAY2}}) in x:
    m.addConstr(x[{{VALUE-NAME}}, {{VALUE-DAY2}}] == 1, "_2")
TYPE: constraint-availability

QUESTION:
What if {{VALUE-NAME}} is also available on {{VALUE-DAY1}} and {{VALUE-DAY2}}.
VALUE-NAME: random.choice(workers)
VALUE-DAY1: random.choice(shifts)
VALUE-DAY2: random.choice(list(set(shifts) - set([{{VALUE-DAY1}}])))
DATA CODE:
availability.append(({{VALUE-NAME}}, {{VALUE-DAY1}}))
availability.append(({{VALUE-NAME}}, {{VALUE-DAY2}}))
TYPE: data-availability

QUESTION:
What if {{VALUE-NAME}} took a promotion and is now paid {{VALUE-COUNT}} dollars per hour?
VALUE-NAME: random.choice(workers)
VALUE-COUNT: pay[{{VALUE-NAME}}] + random.randrange(1,5)
DATA CODE:
pay[{{VALUE-NAME}}] = {{VALUE-COUNT}}
TYPE: data-pay

QUESTION:
What if {{VALUE-NAME}} cannot work more than {{VALUE-COUNT}} shifts?
VALUE-NAME: random.choice(workers)
VALUE-COUNT: random.randrange(3,7)
CONSTRAINT CODE:
m.addConstr(x.sum({{VALUE-NAME}},'*') <= {{VALUE-COUNT}}, "_")
TYPE: shift-limit

QUESTION:
What if I need {{VALUE-COUNT}} additional people on Mondays?
VALUE-COUNT: random.randrange(1,5)
DATA CODE:
shiftRequirements["Mon1"] = shiftRequirements["Mon1"] + {{VALUE-COUNT}}
shiftRequirements["Mon8"] = shiftRequirements["Mon8"] + {{VALUE-COUNT}}
TYPE: data-shift-requirements

QUESTION:
What if I need {{VALUE-COUNT}} people on the weekends?
VALUE-COUNT: random.randrange(2,6)
CONSTRAINT CODE:
shiftRequirements["Sat6"] = {{VALUE-COUNT}}
shiftRequirements["Sun7"] = {{VALUE-COUNT}}
shiftRequirements["Sat13"] = {{VALUE-COUNT}}
shiftRequirements["Sun14"] = {{VALUE-COUNT}}
TYPE: data-shift-requirements

QUESTION:
What if {{VALUE-NAME1}} and {{VALUE-NAME2}} cannot work on the same day?
VALUE-NAME1: random.choice(workers)
VALUE-NAME2: random.choice(list(set(workers) - set([{{VALUE-NAME1}}])))
CONSTRAINT CODE:
m.addConstrs((x[{{VALUE-NAME1}}, s] + x[{{VALUE-NAME2}}, s] <= 1 for s in shifts if ({{VALUE-NAME1}}, s) in x and ({{VALUE-NAME2}}, s) in x), "_")
TYPE: shift-conflicts

QUESTION:
What if all workers cannot do more than {{VALUE-COUNT}} shifts?
VALUE-COUNT: random.randrange(6,10)
CONSTRAINT CODE:
m.addConstrs((x.sum(w,'*') <= {{VALUE-COUNT}} for w in workers), "_")
TYPE: shift-limit

QUESTION:
What if all workers should do at least {{VALUE-COUNT}} shifts?
VALUE-COUNT: random.randrange(1,5)
CONSTRAINT CODE:
m.addConstrs((x.sum(w,'*') >= {{VALUE-COUNT}} for w in workers), "_")
TYPE: shift-limit
