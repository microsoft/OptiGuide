CODE PATH: tsp.py

QUESTION: Why should we go from point {{VALUE-X}} to point {{VALUE-Y}}?
VALUE-IDX: random.randint(0, n - 2)
VALUE-X: tour[{{VALUE-IDX}}]
VALUE-Y: tour[{{VALUE-IDX}} + 1]
CONSTRAINT CODE:
m.addConstr(vars[{{VALUE-X}}, {{VALUE-Y}}] == 0, 'not go')
m.addConstr(vars[{{VALUE-Y}}, {{VALUE-X}}] == 0, 'not go 2')


QUESTION: Can we double the distance between point {{VALUE-X}}  and {{VALUE-Y}}?
VALUE-X: random.randint(0, n - 1)
VALUE-Y: random.choice(list(set(range(n)) - set([{{VALUE-X}}])))
DATA CODE:
if ({{VALUE-X}}, {{VALUE-Y}}) in dist:
    dist[{{VALUE-X}} , {{VALUE-Y}}] *= 2
if ({{VALUE-Y}}, {{VALUE-X}} ) in dist:
    dist[{{VALUE-Y}}, {{VALUE-X}} ] *= 2


QUESTION: what would happen if we remove point {{VALUE-X}}?
VALUE-X: random.randint(0, n - 1)
DATA CODE:
for i, j in list(dist.keys()):
    if i == {{VALUE-X}} or j == {{VALUE-X}}:
        dist[i, j] = 0 # remove the edge cost


QUESTION: What if the edge between point {{VALUE-X}} to {{VALUE-Y}} is removed?
VALUE-IDX: random.randint(0, n - 2)
VALUE-X: tour[{{VALUE-IDX}}]
VALUE-Y: tour[{{VALUE-IDX}} + 1]
CONSTRAINT CODE:
m.addConstr(vars[{{VALUE-X}}, {{VALUE-Y}}] == 0, 'remove')
m.addConstr(vars[{{VALUE-Y}}, {{VALUE-X}}] == 0, 'remove 2')


QUESTION: Can we go from point {{VALUE-X}} to {{VALUE-Y}}?
VALUE-X: random.randint(0, n - 1)
VALUE-Y: random.choice(list(set(range(n)) - set([{{VALUE-X}}])))
CONSTRAINT CODE:
m.addConstr(vars[{{VALUE-X}}, {{VALUE-Y}}] == 1, 'try go')

QUESTION: What if the edge from point {{VALUE-X}} to {{VALUE-Y}} is removed?
VALUE-IDX: random.randint(0, n - 2)
VALUE-X: tour[{{VALUE-IDX}}]
VALUE-Y: tour[{{VALUE-IDX}} + 1]
CONSTRAINT CODE:
m.addConstr(vars[{{VALUE-X}}, {{VALUE-Y}}] == 0, 'remove')
