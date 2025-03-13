
class CoffeeInstance:
    def __init__(self):
        self.capacity_in_supplier = {}
        self.shipping_cost_from_supplier_to_roastery = {}

        self.roasting_cost_light = {}
        self.roasting_cost_dark = {}
        self.shipping_cost_from_roastery_to_cafe = {}

        self.light_coffee_needed_for_cafe = {}
        self.dark_coffee_needed_for_cafe = {}

        self.cafes = []
        self.roasteries = []
        self.suppliers = []

    # Supplier
    def add_supplier(self, supplier, capacity):
        self.capacity_in_supplier[supplier] = capacity
        self.suppliers.append(supplier)

    def update_supplier(self, supplier, capacity):
        if supplier in self.capacity_in_supplier:
            self.capacity_in_supplier[supplier] = capacity

    def delete_supplier(self, supplier):
        if supplier in self.capacity_in_supplier:
            del self.capacity_in_supplier[supplier]
            self.suppliers.remove(supplier)

            # remove all connections to roasteries
            for roastery in self.roasteries:
                self.disconnect_supplier_to_roastery(supplier, roastery)

    # Roastery
    def add_roastery(self, roastery, light_cost, dark_cost):
        self.roasting_cost_light[roastery] = light_cost
        self.roasting_cost_dark[roastery] = dark_cost
        self.roasteries.append(roastery)

    def update_roastery(self, roastery, light_cost=None, dark_cost=None):
        if roastery in self.roasting_cost_light and light_cost != None:
            self.roasting_cost_light[roastery] = light_cost

        if roastery in self.roasting_cost_dark and dark_cost != None:
            self.roasting_cost_dark[roastery] = dark_cost

    def delete_roastery(self, roastery):
        if roastery in self.roasting_cost_light and roastery in self.roasting_cost_dark:
            del self.roasting_cost_light[roastery]
            del self.roasting_cost_dark[roastery]
            self.roasteries.remove(roastery)

            # disconnect all connections to suppliers
            for supplier in self.suppliers:
                self.disconnect_supplier_to_roastery(supplier, roastery)

            # disconnect all connections to cafes
            for cafe in self.cafes:
                self.disconnect_roastery_to_cafe(roastery, cafe)

    # Cafe
    def add_cafe(self, cafe, light_needed, dark_needed):
        self.light_coffee_needed_for_cafe[cafe] = light_needed
        self.dark_coffee_needed_for_cafe[cafe] = dark_needed
        self.cafes.append(cafe)

    def update_cafe(self, cafe, light_needed=None, dark_needed=None):
        if cafe in self.light_coffee_needed_for_cafe and light_needed != None:
            self.light_coffee_needed_for_cafe[cafe] = light_needed

        if cafe in self.dark_coffee_needed_for_cafe and dark_needed != None:
            self.dark_coffee_needed_for_cafe[cafe] = dark_needed

    def delete_cafe(self, cafe):
        if cafe in self.light_coffee_needed_for_cafe and cafe in self.dark_coffee_needed_for_cafe:
            del self.light_coffee_needed_for_cafe[cafe]
            del self.dark_coffee_needed_for_cafe[cafe]
            self.cafes.remove(cafe)

            # disconnect all connections to roasteries
            for roastery in self.roasteries:
                self.disconnect_roastery_to_cafe(roastery, cafe)

    # Supplier -> Roastery
    def get_suppliers_to_roasteries(self):
        return self.shipping_cost_from_supplier_to_roastery.keys()

    def connect_supplier_to_roastery(self, supplier, roastery, shipping_cost):
        self.shipping_cost_from_supplier_to_roastery[(supplier, roastery)] = shipping_cost

    def update_supplier_to_roastery(self, supplier, roastery, shipping_cost):
        if (supplier, roastery) in self.shipping_cost_from_supplier_to_roastery:
            self.shipping_cost_from_supplier_to_roastery[(supplier, roastery)] = shipping_cost

    def disconnect_supplier_to_roastery(self, supplier, roastery):
        if (supplier, roastery) in self.shipping_cost_from_supplier_to_roastery:
            del self.shipping_cost_from_supplier_to_roastery[(supplier, roastery)]

    # Roastery -> Cafe
    def get_roasteries_to_cafes(self):
        return self.shipping_cost_from_roastery_to_cafe.keys()

    def connect_roastery_to_cafe(self, roastery, cafe, shipping_cost):
        self.shipping_cost_from_roastery_to_cafe[(roastery, cafe)] = shipping_cost

    def update_roastery_to_cafe(self, roastery, cafe, shipping_cost):
        if (roastery, cafe) in self.shipping_cost_from_roastery_to_cafe:
            self.shipping_cost_from_roastery_to_cafe[(roastery, cafe)] = shipping_cost

    def disconnect_roastery_to_cafe(self, roastery, cafe):
        if (roastery, cafe) in self.shipping_cost_from_roastery_to_cafe:
            del self.shipping_cost_from_roastery_to_cafe[(roastery, cafe)]
