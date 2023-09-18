CODE PATH: coffee.py

QUESTION:
What would happen if demand at cafe {{VALUE-CAFE}} increased by {{VALUE-NUMBER}}%?
VALUE-CAFE: random.choice(cafes)
VALUE-NUMBER: random.randrange(5,30)
DATA CODE:
light_coffee_needed_for_cafe[{{VALUE-CAFE}}] = light_coffee_needed_for_cafe[{{VALUE-CAFE}}] * (1 + {{VALUE-NUMBER}}/100)
dark_coffee_needed_for_cafe[{{VALUE-CAFE}}] = dark_coffee_needed_for_cafe[{{VALUE-CAFE}}] * (1 + {{VALUE-NUMBER}}/100)
TYPE: demand-increase

QUESTION:
What if demand for light coffee at cafe {{VALUE-CAFE}} increased by {{VALUE-NUMBER}}%?
VALUE-CAFE: random.choice(cafes)
VALUE-NUMBER: random.randrange(5,30)
DATA CODE:
light_coffee_needed_for_cafe[{{VALUE-CAFE}}] = light_coffee_needed_for_cafe[{{VALUE-CAFE}}] * (1 + {{VALUE-NUMBER}}/100)
TYPE: demand-increase-light

QUESTION:
What would happen if the demand at all cafes doubled?
DATA CODE:
for c in cafes:
	light_coffee_needed_for_cafe[c] = light_coffee_needed_for_cafe[c] * 2
	dark_coffee_needed_for_cafe[c] = dark_coffee_needed_for_cafe[c] * 2
TYPE: demand-increase-all

QUESTION:
Why are we using supplier {{VALUE-SUPPLIER}} for roasting facility {{VALUE-ROASTERY}}?
VALUE-SHIPPINGS: [(s, r) for (s, r), value in x.items() if value.X >= 0.999]
VALUE-IDX: random.randint(0, len({{VALUE-SHIPPINGS}}) - 1)
VALUE-SUPPLIER: {{VALUE-SHIPPINGS}}[{{VALUE-IDX}}][0]
VALUE-ROASTERY: {{VALUE-SHIPPINGS}}[{{VALUE-IDX}}][1]
CONSTRAINT CODE:
m.addConstr(x[{{VALUE-SUPPLIER}},{{VALUE-ROASTERY}}] == 0, "_")
TYPE: supply-roastery

QUESTION:
Assume cafe {{VALUE-CAFE}} can exclusively buy coffee from roasting facility {{VALUE-ROASTERY}}, and conversely, roasting facility {{VALUE-ROASTERY}} can only sell its coffee to cafe {{VALUE-CAFE}}. How does that affect the outcome?
VALUE-ROASTERY: random.choice(roasteries)
VALUE-CAFE: random.choice(cafes)
CONSTRAINT CODE:
for c in cafes:
	if c != {{VALUE-CAFE}}:
		m.addConstr(y_light[{{VALUE-ROASTERY}}, c] == 0, "_")
		m.addConstr(y_dark[{{VALUE-ROASTERY}}, c] == 0, "_")
for r in roasteries:
	if r != {{VALUE-ROASTERY}}:
		m.addConstr(y_light[r, {{VALUE-CAFE}}] == 0, "_")
		m.addConstr(y_dark[r, {{VALUE-CAFE}}] == 0, "_")
TYPE: exclusive-roastery-cafe

QUESTION:
What if roasting facility {{VALUE-ROASTERY}} can only be used for cafe {{VALUE-CAFE}}?
VALUE-ROASTERY: random.choice(roasteries)
VALUE-CAFE: random.choice(cafes)
CONSTRAINT CODE:
for c in cafes:
	if c != {{VALUE-CAFE}}:
		m.addConstr(y_light[{{VALUE-ROASTERY}}, c] == 0, "_")
		m.addConstr(y_dark[{{VALUE-ROASTERY}}, c] == 0, "_")
TYPE: incompatible-roastery-cafes

QUESTION:
What if supplier {{VALUE-SUPPLIER}} can now provide only half of the quantity?
VALUE-SUPPLIER: random.choice(suppliers)
DATA CODE:
capacity_in_supplier[{{VALUE-SUPPLIER}}] = capacity_in_supplier[{{VALUE-SUPPLIER}}]/2
TYPE: supplier-capacity

QUESTION:
The per-unit cost from supplier {{VALUE-SUPPLIER}} to roasting facility {{VALUE-ROASTERY}} is now {{VALUE-NUMBER}}. How does that affect the total cost?
VALUE-SUPPLIER: random.choice(suppliers)
VALUE-ROASTERY: random.choice(roasteries)
VALUE-NUMBER: random.randrange(1,10)
DATA CODE:
shipping_cost_from_supplier_to_roastery[{{VALUE-SUPPLIER}},{{VALUE-ROASTERY}}] = {{VALUE-NUMBER}}
TYPE: supplier-roastery-shipping

QUESTION:
What would happen if roastery 2 produced at least as much light coffee as roastery 1?
CONSTRAINT CODE:
m.addConstr(sum(y_light['roastery1',c] for c in cafes) <= sum(y_light['roastery2',c] for c in cafes), "_")
TYPE: light-quantities-roasteries

QUESTION:
What would happen if roastery 1 produced less light coffee than roastery 2?
CONSTRAINT CODE:
m.addConstr(sum(y_light['roastery1',c] for c in cafes) <= sum(y_light['roastery2',c] for c in cafes) - 1, "_")
TYPE: light-quantities-roasteries

QUESTION:
What will happen if supplier 1 ships more to roastery 1 than roastery 2?
CONSTRAINT CODE:
m.addConstr(x['supplier1','roastery1'] >= x['supplier1','roastery2'] + 1, "_")
TYPE: shipping-quantities-roasteries

QUESTION:
What will happen if supplier 1 ships to roastery 1 at least as much as to roastery 2?
CONSTRAINT CODE:
m.addConstr(x['supplier1','roastery1'] >= x['supplier1','roastery2'], "_")
TYPE: shipping-quantities-roasteries

QUESTION:
Why not only use a single supplier for roastery 2?
CONSTRAINT CODE:
z = m.addVars(suppliers, vtype=GRB.BINARY, name="z")
m.addConstr(sum(z[s] for s in suppliers) <= 1, "_")
for s in suppliers:
	m.addConstr(x[s,'roastery2'] <= capacity_in_supplier[s] * z[s], "_")
TYPE: single-supplier-roastery
