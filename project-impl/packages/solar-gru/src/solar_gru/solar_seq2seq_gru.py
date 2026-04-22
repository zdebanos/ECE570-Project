import torch
import torch.nn as nn
import numpy as np
import random

from typing import Final, List, Any, Optional

_N_FEATURES:       Final = 11
_HIDDEN_SIZE:      Final = 64
_DECODER_FEATURES: Final = 7
_OUTPUT_SIZE:      Final = 1
_CONSECUTIVE_HOURS_AFTER: Final = 48

class SolarSeq2SeqGRU(torch.nn.Module):
    def __init__(self, weights=None):
        # The encoder inputs are: time (4 scalars), GTI(i-1), weather variables.
        # The decoder inputs are: time (4 scalars), GTI(i-1), temp and cloud_cover
        super().__init__()
        self.encoder = nn.GRU(input_size=_N_FEATURES,       hidden_size=_HIDDEN_SIZE, batch_first=True)
        self.decoder = nn.GRU(input_size=_DECODER_FEATURES, hidden_size=_HIDDEN_SIZE, batch_first=True)
        self.fc      = nn.Linear(_HIDDEN_SIZE,  _OUTPUT_SIZE)
        self.set_weights(weights)
    
    def set_weights(self, weights):
        """Sets the model's weights.
        """
        self._weights = weights
        if self._weights is None:
            return
        self.load_state_dict(weights)
        
    def forward(
        self,
        prior,
        teacher_seq,
        device,
        cur_epoch=0,
        max_epoch=100, 
        target_len=_CONSECUTIVE_HOURS_AFTER
    ) -> torch.Tensor:
        """Seq2seq forward method.

        Args:
            prior:       Encoder data.
            teacher_seq: Decoder data.
            device:      Torch device.
            cur_epoch:   Current training epoch. It's used to turn off the teacher forcing
                         during the training phase.
            max_epoch:   Specify the maximum epoch in training.
            target_len:  The decoder prediction length.
        """
        prior = prior.to(device)
        teacher_seq = teacher_seq.to(teacher_seq)
        self.to(device)

        all_preds: List[torch.Tensor] = []
        _, hidden  = self.encoder(prior)
        
        if self.training: # Training with teacher forcing.
            # When the epoch threshold is less than 0.7, we use teacher forcing only.
            # This should speed up the training as we don't iterate.
            use_teacher_forcing = True
            if cur_epoch > 0.7 * max_epoch:
                if random.random() < 0.2:
                    use_teacher_forcing = False

            if use_teacher_forcing:
                decoder_out, _ = self.decoder(teacher_seq, hidden)
                preds = self.fc(decoder_out)
                return preds.squeeze(-1)
            else:
                curr_input = teacher_seq[:, 0:1, :]
                for i in range(target_len):
                    out, hidden = self.decoder(curr_input, hidden)
                    pred = self.fc(out)
                    all_preds.append(pred)
                    if i < target_len - 1:
                        if use_teacher_forcing:
                            curr_input = teacher_seq[:, i+1:i+2, :]
                        else:
                            curr_input: torch.Tensor = torch.cat([
                                teacher_seq[:, i+1:i+2, 0:4],
                                pred,
                                teacher_seq[:, i+1:i+2, 5:7]], dim=2)
                return torch.cat(all_preds, dim=1).squeeze(-1)

        # Evaluation mode
        else:
            # We don't need to teacher forcing GTI data, but we need
            # the timestamps. The first timestamp can be directly taken from teacher_seq.
            # Don't forget to keep the batch_size.
            curr_input = teacher_seq[:, 0:1, 0:7]
            for i in range(target_len):
                out, hidden = self.decoder(curr_input, hidden)
                pred = self.fc(out)
                all_preds.append(pred)
                if i < target_len-1:
                    curr_input = torch.cat([teacher_seq[:, i+1:i+2, 0:4], pred, teacher_seq[:, i+1:i+2, 5:7]], dim=2)
            return torch.cat(all_preds, dim=1).squeeze(-1)

    @torch.inference_mode()
    def forecast(
        self,
        prior: torch.Tensor,               # Use normalized data
        forecast_meteo_data: torch.Tensor, # Use normalized data
        device: torch.device
    ) -> np.ndarray | None:
        """This function passes the `prior` data through the encoder,
        then `forecast_meteo_data` is passed through the decoder, using then
        previous encoder hidden state.

        Args:
            prior:               The previous days data (use normalized data).
            forecast_meteo_data: Two days of additional information (forecasted),
                                 use normalized data. Use a tensor of size (bs, 8),
                                 where bs is the batch_size. The 5th element (GTI(i-1))
                                 is always ignored and therefore be of any value.
            device:              Torch device.

        Returns:
            The two days of predicted GTI(i) data [W].
        """
        if self._weights is None:
            # No weights are present, throw an error
            return None
        self.eval()
        with torch.no_grad():
            return self(prior, forecast_meteo_data, device).cpu().numpy()[0]
