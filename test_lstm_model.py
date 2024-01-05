import torch
import unittest
from LSTM import LSTM
class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        self.output_dim = 5
        self.batch_size = 3
        self.seq_length = 7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim).to(self.device)

    def test_output_dimension(self):
        input_data = torch.rand(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        output = self.model(input_data)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_forward_pass(self):
        input_data = torch.rand(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        try:
            _ = self.model(input_data)
        except Exception as e:
            self.fail(f"Forward pass failed with exception {e}")

    def test_batch_processing(self):
        input_data = torch.rand(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        output = self.model(input_data)
        self.assertEqual(output.shape[0], self.batch_size)


    def test_device_assignment(self):
        device = next(self.model.parameters()).device
        self.assertTrue(device == torch.device('cpu') or device.type == 'cuda')

if __name__ == '__main__':
    unittest.main()
