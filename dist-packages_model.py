1 from abc import ABC, abstractmethod
2 from typing import TYPE_CHECKING, Any
3 
4 import torch
5 from torch import nn
6 
7 from trainer._types import ValueListDict
8 
9 if TYPE_CHECKING:
10     from trainer.trainer import Trainer
11 
12 
13 class TrainerModel(ABC, nn.Module):
14     """Abstract 🐸TTS class. Every new 🐸TTS model must inherit this."""
15 
16     @abstractmethod
17     def forward(
18         self, input: torch.Tensor, *args: Any, aux_input: dict[str, Any] | None = None, **kwargs: Any
19     ) -> dict[str, Any]:
20         """Forward ... for the model mainly used in training.
21 
22         You can be flexible here and use different number of arguments and argument names since it is intended to be
23         used by `train_step()` without exposing it out of the model.
24 
25         Args:
26             input (torch.Tensor): Input tensor.
27             aux_input (Dict): Auxiliary model inputs like embeddings, durations or any other sorts of inputs.
28 
29         Returns:
30             Dict: Model outputs. Main model output must be named as "model_outputs".
31         """
32         if aux_input is None:
33             aux_input = {}
34         outputs_dict = {"model_outputs": None}
35         ...
36         return outputs_dict
37 
38     def format_batch(self, batch: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
39         """Format batch returned by the data loader before sending it to the model.
40 
41         If not implemented, model uses the batch as is.
42         Can be used for data augmentation, feature ectraction, etc.
43         """
44         return batch
45 
46     def format_batch_on_device(self, batch: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
47         """Format batch on device before sending it to the model.
48 
49         If not implemented, model uses the batch as is.
50         Can be used for data augmentation, feature ectraction, etc.`
51         """
52         return batch
53 
54     def train_step(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
55         """Perform a single training step. Run the model forward ... and compute losses.
56 
57         Args:
58             batch (Dict): Input tensors.
59             criterion (nn.Module): Loss layer designed for the model.
60             optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.
61 
62         Returns:
63             Tuple[Dict, Dict]: Model outputs and computed losses.
64         """
65         msg = " [!] `train_step()` is not implemented."
66         raise NotImplementedError(msg)
67 
68     def train_log(self, *args: Any, **kwargs: Any) -> None:
69         """Create visualizations and waveform examples for training.
70 
71         For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
72         be projected onto Tensorboard.
73 
74         Args:
75             batch (Dict): Model inputs used at the previous training step.
76             outputs (Dict): Model outputs generated at the previoud training step.
77             logger (Logger): Logger instance to log training plots.
78             assets (Dict): Assets to be used for logging from the trainer's closure.
79             steps (int): Number of training steps taken so far.
80 
81         Returns:
82             Tuple[Dict, np.ndarray]: training plots and output waveform.
83         """
84         msg = " [!] `train_log()` is not implemented."
85         raise NotImplementedError(msg)
86 
87     @torch.inference_mode()
88     def eval_step(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
89         """Perform a single evaluation step.
90 
91         Run the model forward ... and compute losses. In most cases, you can
92         call `train_step()` with no changes.
93 
94         Args:
95             batch (Dict): Input tensors.
96             criterion (nn.Module): Loss layer designed for the model.
97             optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.
98 
99         Returns:
100             Tuple[Dict, Dict]: Model ouputs and computed losses.
101         """
102         msg = " [!] `eval_step()` is not implemented."
103         raise NotImplementedError(msg)
104 
105     def eval_log(self, *args: Any, **kwargs: Any) -> None:
106         """The same as `train_log()`."""
107         msg = " [!] `eval_log()` is not implemented."
108         raise NotImplementedError(msg)
109 
110     @abstractmethod
111     def get_data_loader(*args: Any, **kwargs: Any) -> torch.utils.data.DataLoader[Any]:
112         """Get data loader for the model.
113 
114         Args:
115             config (TrainerConfig): Configuration object.
116             assets (Dict): Additional assets to be used for data loading.
117             is_eval (bool): If True, returns evaluation data loader.
118             samples (Union[List[Dict], List[List]]): List of samples to be used for data loading.
119             verbose (bool): If True, prints data loading information.
120             num_gpus (int): Number of GPUs used for training.
121             rank (int): Rank of the current GPU.
122 
123         Returns:
124             torch.utils.data.DataLoader: Data loader for the model.
125         """
126         ...
127         msg = " [!] `get_data_loader()` is not implemented."
128         raise NotImplementedError(msg)
129 
130     def get_train_data_loader(*args: Any, **kwargs: Any) -> torch.utils.data.DataLoader[Any]:
131         raise NotImplementedError
132 
133     def get_eval_data_loader(*args: Any, **kwargs: Any) -> torch.utils.data.DataLoader[Any]:
134         raise NotImplementedError
135 
136     def get_test_data_loader(*args: Any, **kwargs: Any) -> torch.utils.data.DataLoader[Any]:
137         raise NotImplementedError
138 
139     def test_run(self, *args: Any, **kwargs: Any):
140         raise NotImplementedError
141 
142     def test(self, assets: dict[str, Any], data_loader: torch.utils.data.DataLoader[Any], outputs: Any | None = None):
143         raise NotImplementedError
144 
145     def test_log(self, *args: Any, **kwargs: Any):
146         raise NotImplementedError
147 
148     def init_for_training(self) -> None:
149         """Initialize model for training."""
150 
151     def optimize(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
152         """Model specific optimization step that must perform the following steps.
153 
154             1. Forward pass
155             2. Compute loss
156             3. Backward pass
157             4. Update weights.
158 
159         Use `self.scaled_backward()` instead of `loss.backward()` to be able to use Mixed Precision Training.
160 
161         Args:
162             batch (Dict): Input tensors.
163             trainer (Trainer): Trainer instance to be able to access the training closure.
164 
165         Returns:
166             Tuple[Dict, Dict, float]: Model outputs, loss dictionary.
167         """
168         msg = " [!] `optimize()` is not implemented."
169         raise NotImplementedError(msg)
170 
171     def scaled_backward(
172         self,
173         loss: torch.Tensor,
174         trainer: "Trainer",
175         *args: Any,
176         **kwargs: Any,
177     ) -> None:
178         """Backward pass with gradient scaling for custom `optimize` calls.
179 
180         Args:
181             loss (torch.Tensor): Loss to be backpropagated.
182             trainer (Trainer): Trainer instance to be able to access the training closure.
183         """
184         if trainer.use_amp_scaler:
185             if trainer.scaler is not None:
186                 # model optimizer step in mixed precision mode
187                 trainer.scaler.scale(loss).backward()
188         else:
189             # main model optimizer step
190             loss.backward()
191 
192     def get_optimizer(self) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
193         """Setup an return optimizer or optimizers."""
194         raise NotImplementedError
195 
196     def get_lr(self) -> float | list[float]:
197         """Return learning rate(s).
198 
199         Returns:
200             Union[float, List[float]]: Model's initial learning rates.
201         """
202         raise NotImplementedError
203 
204     def get_scheduler(
205         self, optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | dict[str, torch.optim.Optimizer]
206     ):
207         raise NotImplementedError
208 
209     def get_criterion(self) -> nn.Module | list[nn.Module]:
210         """Return model criterion."""
211         msg = "`get_criterion` is not implemented."
212         raise NotImplementedError(msg)
213 
214     ## Callbacks
215     def on_init_start(self, trainer: "Trainer") -> None: ...
216 
217     def on_init_end(self, trainer: "Trainer") -> None: ...
218 
219     def on_epoch_start(self, trainer: "Trainer") -> None: ...
220 
221     def on_epoch_end(self, trainer: "Trainer") -> None: ...
222 
223     def on_train_epoch_start(self, trainer: "Trainer") -> None: ...
224 
225     def on_train_epoch_end(self, trainer: "Trainer") -> None: ...
226 
227     @staticmethod
228     def before_backward_pass(loss_dict: dict[str, Any], optimizer: ValueListDict[torch.optim.Optimizer]) -> None: ...
229 
230     @staticmethod
231     def before_gradient_clipping() -> None: ...
232 
233     def on_train_step_start(self, trainer: "Trainer") -> None: ...
234 
235     def on_train_step_end(self, trainer: "Trainer") -> None: ...
236 
237     def on_keyboard_interrupt(self, trainer: "Trainer") -> None: ...