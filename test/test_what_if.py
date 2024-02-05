import unittest

from optiguide.optiguide import OptiGuideAgent, _insert_code

PYOMO_SRC_CODE = open("benchmark/application/coffee_pyomo.py").read()
GUROBI_SRC_CODE = open("benchmark/application/coffee.py").read()

class TestGurobiWhatIf(unittest.TestCase):

    def setUp(self):
        # Setting up a basic environment for testing
        self.source_code = "some_sample_source_code_here"
        self.agent = OptiGuideAgent(name="TestAgent", 
                                    source_code=GUROBI_SRC_CODE, 
                                    solver_software="gurobi")
        import pdb; pdb.set_trace()
        
    def test_initialization(self):
        """Test initialization of OptiGuideAgent."""
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent._solver_software, "gurobi")

    def test_generate_reply_non_writer_safeguard(self):
        """Test generate_reply method for non-writer and non-safeguard agents."""
        response = self.agent.generate_reply(sender=None)
        self.assertIsInstance(response, str)

    def test_insert_code_gurobi(self):
        """Test _insert_code method for gurobi solver."""
        new_lines = "m.addConstr(x + y <= 10)"
        updated_code = _insert_code(self.source_code, new_lines, "gurobi")
        self.assertIn(new_lines, updated_code)


class TestPyomoWhatIf(unittest.TestCase):
    def setUp(self):
        # Setting up a basic environment for testing
        self.source_code = "some_sample_source_code_here"
        self.agent = OptiGuideAgent(name="TestAgent", 
                                    source_code=PYOMO_SRC_CODE, 
                                    solver_software="gurobi")

    def test_initialization(self):
        """Test initialization of OptiGuideAgent."""
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent._solver_software, "gurobi")

    def test_generate_reply_non_writer_safeguard(self):
        """Test generate_reply method for non-writer and non-safeguard agents."""
        response = self.agent.generate_reply(sender=None)
        self.assertIsInstance(response, str)

    def test_insert_code_pyomo(self):
        """Test _insert_code method for pyomo solver."""
        new_lines = "model.Constraint(expr = x + y <= 10)"
        updated_code = _insert_code(self.source_code, new_lines, "pyomo")
        self.assertIn(new_lines, updated_code)

    # Add more test cases as needed to cover different functionalities

if __name__ == '__main__':
    unittest.main()
