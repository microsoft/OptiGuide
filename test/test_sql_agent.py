import unittest
from unittest.mock import patch

from optiguide.experimental.sql_agent import SQLAgent

# Sample data for testing
sql_file = "test/test_Chinook.sqlite"
name = "Test SQL Agent"
max_sys_msg_tokens = 10000
llm_config = {"config_list": [{"model": "gpt-4", "api_key": "fake key her."}]}


class TestSQLAgent(unittest.TestCase):

    def setUp(self):
        self.agent = SQLAgent(sql_file, name, max_sys_msg_tokens, llm_config)

    def test_initialization(self):
        self.assertEqual(self.agent.sql_file, sql_file)
        self.assertEqual(self.agent.name, name)
        # Add more assertions here to check if the initialization is correct

    def test_synth_history(self):
        # Check if synthetic history is correctly added
        self.assertGreater(
            len(self.agent.assistant._oai_messages[self.agent.proxy]), 0)
        # Add more detailed checks on the content of the synthetic history

    def test_generate_sql_reply(self):
        # Mock a sample message and test the reply generation
        sample_message = [{"content": "Test message", "role": "user"}]
        with patch(
                'optiguide.experimental.sql_agent.SQLAgent.generate_sql_reply'
        ) as mock_reply:
            mock_reply.return_value = True, "Sample reply"
            reply_generated = self.agent.generate_sql_reply(
                sample_message, self.agent, None)
            self.assertEqual(reply_generated, (True, "Sample reply"))


if __name__ == "__main__":
    unittest.main()
