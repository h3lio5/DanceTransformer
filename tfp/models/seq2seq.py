"""Sequence-to-sequence model for human motion prediction."""
import random
import torch
from torch import nn


class Seq2SeqModel(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 architecture,
                 rnn_size,
                 num_layers=1,
                 num_joints=21,
                 residual_velocities=False,
                 dropout=0.0,
                 teacher_ratio=0.0):
        """Create the model.
        Args:
          architecture: [basic, tied] whether to tie the encoder and decoder.
          rnn_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          residual_velocities: whether to use a residual connection that models velocities.
        """
        super(Seq2SeqModel, self).__init__()

        self.input_size = num_joints*3
        # Summary writers for train and test runs
        self.rnn_size = rnn_size
        self.dropout = nn.Dropout(dropout)
        self.residual_velocities = residual_velocities
        # === Create the RNN that will keep the state ===
        self.encoder = torch.nn.GRU(
            self.input_size, self.rnn_size, batch_first=True, num_layers=num_layers)
        self.decoder = torch.nn.GRUCell(
            self.rnn_size, self.rnn_size)

        self.projector = nn.Linear(self.rnn_size, self.input_size)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        Args:
            encoder_inputs: batch of dance pose sequences , shape=(batch_size,source_seq_length,num_joints*3)
            decoder_inputs: batch of dance pose sequences , shape=(batch_size,target_seq_length-1,num_joints*3)
        Returns:
            outputs: batch of predicted dance pose sequences, shape=(batch_size,target_seq_length-1,num_joints*3)
        """
        # First calculate the encoder hidden state
        all_hidden_states, encoder_hidden_state = self.encoder(
            encoder_inputs)

        outputs = []
        next_state = encoder_hidden_state
        # Iterate over decoder inputs
        for inp in decoder_inputs:
            # Perform teacher forcing
            if random.random() < self.teacher_forcing:
                inp = prev_output
            next_state = self.decoder(inp, next_state)
            # Apply residual network to help in smooth transition between subsequent poses
            if self.residual_velocities:
                output = inp + self.projector(self.dropout(next_state))
            else:
                output = self.projector(self.dropout(next_state))
            # Store the output for Teacher Forcing: use the prediction as
            # the next input instead of feeding the ground truth
            prev_output = output
            outputs.append(output.view(
                [1, decoder_inputs.size(0), self.input_size]))

        outputs = torch.cat(outputs, 0)
        return torch.transpose(outputs, 0, 1)
