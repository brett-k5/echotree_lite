1 import functools
2 import gc
3 import logging
4 import os
5 import platform
6 import shutil
7 import sys
8 import time
9 import traceback
10 from collections.abc import Callable, Generator
11 from contextlib import nullcontext, suppress
12 from inspect import signature
13 from pathlib import Path
14 from typing import Any, Optional, cast, overload
15 
16 import torch
17 import torch.distributed as dist
18 from torch import nn
19 from torch.nn.parallel import DistributedDataParallel as DDP_th
20 from torch.utils.data import DataLoader
21 
22 from trainer._types import Callback, LossDict, LRScheduler, ValueListDict
23 from trainer.callbacks import TrainerCallback
24 from trainer.config import TrainerArgs, TrainerConfig
25 from trainer.generic_utils import (
26     KeepAverage,
27     count_parameters,
28     get_experiment_folder_path,
29     get_git_branch,
30     is_pytorch_at_least_2_3,
31     is_pytorch_at_least_2_4,
32     iter_value_list_dict,
33     map_value_list_dict,
34     remove_experiment_folder,
35     set_partial_state_dict,
36     to_cuda,
37 )
38 from trainer.io import (
39     copy_model_files,
40     get_last_checkpoint,
41     load_fsspec,
42     save_best_model,
43     save_checkpoint,
44 )
45 from trainer.logging import ConsoleLogger, DummyLogger, logger_factory
46 from trainer.logging.base_dash_logger import BaseDashboardLogger
47 from trainer.model import TrainerModel
48 from trainer.trainer_utils import (
49     get_optimizer,
50     get_scheduler,
51     print_training_env,
52     setup_torch_training_env,
53 )
54 from trainer.utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
55 from trainer.utils.distributed import (
56     get_rank,
57     init_distributed,
58     rank_zero_logger_info,
59     rank_zero_only,
60 )
61 
62 logger = logging.getLogger("trainer")
63 
64 if is_pytorch_at_least_2_3():
65     GradScaler = functools.partial(torch.GradScaler, device="cuda")
66 else:
67     GradScaler = torch.cuda.amp.GradScaler  # type: ignore[assignment]
68 
69 
70 class Trainer:
71     def __init__(  # pylint: disable=dangerous-default-value
72         self,
73         args: TrainerArgs,
74         config: TrainerConfig,
75         output_path: str | os.PathLike[Any] | None = None,
76         *,
77         c_logger: ConsoleLogger | None = None,
78         dashboard_logger: BaseDashboardLogger | None = None,
79         model: TrainerModel | None = None,
80         get_model: Callable[..., TrainerModel] | None = None,
81         get_data_samples: Callable[..., list[Any]] | None = None,
82         train_samples: list[Any] | None = None,
83         eval_samples: list[Any] | None = None,
84         test_samples: list[Any] | None = None,
85         train_loader: DataLoader[Any] | None = None,
86         eval_loader: DataLoader[Any] | None = None,
87         training_assets: dict[str, Any] | None = None,
88         parse_command_line_args: bool = True,
89         callbacks: dict[str, Callback] | None = None,
90         gpu: int | None = None,
91     ) -> None:
92         """Simple yet powerful 🐸💬 TTS trainer for PyTorch.
93 
94         It can train all the available `tts` and `vocoder` models or easily be customized.
95 
96         Notes:
97             Supports Automatic Mixed Precision training using PyTorch's native `amp` module.
98 
99         Args:
100             args (TrainerArgs): Training arguments parsed either from console by `argparse` or `TrainerArgs`
101                 config object.
102 
103             config (TrainerConfig): Model config object. It includes all the values necessary for initializing, training, evaluating
104                 and testing the model.
105 
106             output_path (str or Path, optional): Path to the output training folder. All
107                 the files are saved under this path. Uses value from config if None.
108 
109             c_logger (ConsoleLogger, optional): Console logger for printing training status. If not provided, the default
110                 console logger is used. Defaults to None.
111 
112             dashboard_logger Union[TensorboardLogger, WandbLogger]: Dashboard logger. If not provided, the tensorboard logger is used.
113                 Defaults to None.
114 
115             model (TrainerModel, optional): Initialized and ready-to-train model. If it is not defined, `Trainer`
116                 initializes a model from the provided config. Defaults to None.
117 
118             get_model (Callable):
119                 A function that returns a model. It is used to initialize the model when `model` is not provided.
120                 It either takes the config as the only argument or does not take any argument.
121                 Defaults to None
122 
123             get_data_samples (Callable):
124                 A function that returns a list of training and evaluation samples. Used if `train_samples` and
125                 `eval_samples` are None. Defaults to None.
126 
127             train_samples (List):
128                 A list of training samples used by the model's `get_train_data_loader` to init the `dataset` and the
129                 `data_loader`. Defaults to None.
130 
131             eval_samples (List):
132                 A list of evaluation samples used by the model's `get_eval_data_loader` to init the `dataset` and the
133                 `data_loader`. Defaults to None.
134 
135             train_loader (DataLoader):
136                 A pytorch data loader object for training epochs. Leave as None if you want it to be made during training. Defaults to None.
137 
138             eval_loader (DataLoader):
139                 A pytorch data loader object for evaluation epochs. Leave as None to be generated during training. Defaults to None.
140 
141             test_samples (List):
142                 A list of test samples used by the model's `get_test_data_loader` to init the `dataset` and the
143                 `data_loader`. If None, the ```model.test_run()``` is expected to load the data. Defaults to None.
144 
145             training_assets (Dict):
146                 A dictionary of assets to be used at training and passed to the model's ```train_log(), eval_log(), get_data_loader()```
147                 during training. It can include  `AudioProcessor` or/and `Tokenizer`. Defaults to {}.
148 
149             parse_command_line_args (bool):
150                 If true, parse command-line arguments and update `TrainerArgs` and model `config` values. Set it
151                 to false if you parse the arguments yourself. Defaults to True.
152 
153             callbacks (Dict[str, Callable]):
154                 A dictionary of callbacks to be used during training. The keys are the callback names and the values
155 
156             gpu (int):
157                 GPU ID to use for training If "CUDA_VISIBLE_DEVICES" is not set. Defaults to None.
158 
159         Example::
160 
161             Running trainer with a model.
162 
163             >>> args = TrainerArgs(...)
164             >>> config = ModelConfig(...)
165             >>> model = Model(config)
166             >>> trainer = Trainer(args, config, model=model)
167             >>> trainer.fit()
168 
169         TODO:
170                 - Wrap model for not calling .module in DDP.
171                 - Deepspeed integration
172                 - Profiler integration.
173                 - Overfitting to a batch.
174                 - TPU training
175         """
176         if training_assets is None:
177             training_assets = {}
178         if callbacks is None:
179             callbacks = {}
180 
181         if parse_command_line_args:
182             # parse command-line arguments to override TrainerArgs()
183             coqpit_overrides = args.parse_known_args(arg_prefix="")
184 
185             # get ready for training and parse command-line arguments to override the model config
186             config, new_fields = self.init_training(args, coqpit_overrides, config)
187         elif args.continue_path or args.restore_path:
188             config, new_fields = self.init_training(args, [], config)
189         else:
190             new_fields = {}
191 
192         # set the output path
193         if args.continue_path:
194             self.continue_run = True
195             # use the same path as the continuing run
196             output_path = args.continue_path
197         else:
198             self.continue_run = False
199             # override the output path if it is provided
200             output_path = config.output_path if output_path is None else str(output_path)
201             # create a new output folder name
202             output_path = get_experiment_folder_path(output_path, config.run_name)
203             output_path.mkdir(exist_ok=True, parents=True)
204 
205         # copy training assets to the output folder
206         copy_model_files(config, output_path, new_fields)
207 
208         # init class members
209         self.args = args
210         self.config = config
211         self.output_path = Path(output_path)
212         self.training_assets = training_assets
213         self.grad_accum_steps = args.grad_accum_steps
214         self.overfit_batch = args.overfit_batch
215         self.skip_train_epoch = args.skip_train_epoch
216         self.start_with_eval = args.start_with_eval
217 
218         assert self.grad_accum_steps > 0, " [!] grad_accum_steps must be greater than 0."
219 
220         # setup logging
221         log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
222         self._setup_logger_config(log_file)
223 
224         # setup training environment
225         self.use_cuda, self.num_gpus = self.setup_training_environment(args=args, config=config, gpu=gpu)
226 
227         # init loggers
228         self.dashboard_logger, self.c_logger = self.init_loggers(self.config, output_path, dashboard_logger, c_logger)
229         # self.c_logger.logger = logger
230 
231         self.log_model_step = (
232             self.config.log_model_step if self.config.log_model_step is not None else self.config.save_step
233         )
234 
235         # make sure that start_with_eval is disabled if eval is disabled
236         if not self.config.run_eval and self.start_with_eval:
237             self.start_with_eval = False
238 
239         self.total_steps_done = 0
240         self.epochs_done = 0
241         self.best_loss: LossDict | float = {
242             "train_loss": float("inf"),
243             "eval_loss": float("inf") if self.config.run_eval else None,
244         }
245         self.train_loader: DataLoader[Any] | None = None
246         self.test_loader: DataLoader[Any] | None = None
247         self.eval_loader: DataLoader[Any] | None = None
248 
249         self.keep_avg_train: KeepAverage | None = None
250         self.keep_avg_eval: KeepAverage | None = None
251 
252         self.use_amp_scaler = (
253             self.use_cuda
254             if self.config.mixed_precision and self.config.precision == "fp16"
255             else self.config.use_grad_scaler
256         )
257 
258         self.train_samples: list[Any] | None
259         self.eval_samples: list[Any] | None
260         self.test_samples: list[Any] | None
261         if train_samples is not None:
262             # use the provided samples
263             self.train_samples = train_samples
264             self.eval_samples = eval_samples
265             self.test_samples = test_samples
266         elif get_data_samples is not None:
267             # run `get_data_samples` to init the data samples
268             (
269                 self.train_samples,
270                 self.eval_samples,
271                 self.test_samples,
272             ) = self.run_get_data_samples(config, get_data_samples)
273         else:
274             # expecting to load the samples in `model.get_data_loader()`
275             self.train_samples = None
276             self.eval_samples = None
277             self.test_samples = None
278 
279         # define custom train and eval loader
280         self.train_loader = train_loader
281         self.eval_loader = eval_loader
282 
283         # only use a subset of the samples if small_run is set
284         self.setup_small_run(args.small_run)
285 
286         # init the model
287         if model is not None:
288             self.model = model
289         elif get_model is not None:
290             self.run_get_model(self.config, get_model)
291         else:
292             msg = "`model` and `get_model` cannot both be None."
293             raise ValueError(msg)
294 
295         # init model's training assets
296         self.model.init_for_training()
297 
298         # setup criterion
299         self.criterion = self.get_criterion(self.model)
300 
301         # DISTRUBUTED
302         if self.use_pt_ddp:
303             rank_zero_logger_info(" > Using PyTorch DDP", logger)
304             init_distributed(
305                 args.rank,
306                 self.num_gpus,
307                 args.group_id,
308                 self.config.distributed_backend,
309                 self.config.distributed_url,
310             )
311 
312         if self.use_cuda:
313             self.model.cuda()
314             if isinstance(self.criterion, list):
315                 for criterion in self.criterion:
316                     if isinstance(criterion, nn.Module):
317                         criterion.cuda()
318             elif isinstance(self.criterion, nn.Module):
319                 self.criterion.cuda()
320 
321         # setup optimizer and scheduler
322         self.optimizer = self.get_optimizer(self.model, self.config)
323         self.scheduler = self.get_scheduler(self.model, self.config, self.optimizer)
324         # With multiple optimizers, some are not used all the time. We keep
325         # track of that to know whether to step the corresponding schedulers.
326         self._stepped_optimizers: set[int | None] = set()
327 
328         # CALLBACK
329         self.callbacks = TrainerCallback()
330         self.callbacks.parse_callbacks_dict(callbacks)
331         self.callbacks.on_init_start(self)
332 
333         # init AMP
334         self.scaler = GradScaler() if self.use_amp_scaler else None
335 
336         # restore model
337         if self.args.restore_path:
338             self.restore_model()
339 
340         # DISTRIBUTED
341         self.wrapped_model: TrainerModel | None = None
342         if self.use_pt_ddp:
343             ddp_model = DDP_th(self.model, device_ids=[args.rank], output_device=args.rank)
344             self.wrapped_model = ddp_model.module  # cast(TrainerModel, ddp_model.module)
345 
346         # setup accelerator
347         self.setup_accelerate()
348 
349         # count model size
350         num_params = count_parameters(self.model)
351         rank_zero_logger_info(f"\n > Model has {num_params} parameters", logger)
352 
353         self.callbacks.on_init_end(self)
354         self.dashboard_logger.add_config(config)
355         self.save_training_script()
356 
357     @property
358     def use_pt_ddp(self) -> bool:
359         """Return True if using PyTorch DDP."""
360         return self.num_gpus > 1 and not self.use_accelerate
361 
362     @property
363     def use_accelerate(self) -> bool:
364         """Return True if using HF Accelerate."""
365         return self.args.use_accelerate
366 
367     def setup_accelerate(self) -> None:
368         if self.use_accelerate:
369             self.model, self.optimizer, self.train_loader, self.scheduler, self.accelerator = self.init_accelerate(
370                 model=self.model,
371                 optimizer=self.optimizer,
372                 training_dataloader=self.train_loader,
373                 scheduler=self.scheduler,
374                 grad_accum_steps=self.grad_accum_steps,
375                 mixed_precision=self.config.mixed_precision,
376                 precision=self.config.precision,
377             )
378 
379     def prepare_accelerate_loader(self, data_loader: DataLoader[Any]) -> DataLoader[Any]:
380         """Prepare the accelerator for the training."""
381         if self.use_accelerate:
382             return self.accelerator.prepare_data_loader(data_loader)
383         return data_loader
384 
385     @staticmethod
386     def init_accelerate(
387         model: TrainerModel,
388         optimizer: ValueListDict[torch.optim.Optimizer],
389         training_dataloader: DataLoader[Any] | None,
390         scheduler: LRScheduler | list[LRScheduler] | dict[str, LRScheduler] | None,
391         *,
392         grad_accum_steps: int,
393         mixed_precision: bool,
394         precision: str,
395     ) -> tuple:
396         """Setup HF Accelerate for the training."""
397         # check if accelerate is installed
398         try:
399             from accelerate import Accelerator  # pylint:disable=import-outside-toplevel
400         except ImportError as e:
401             msg = "Please install accelerate to use this feature."
402             raise ImportError(msg) from e
403 
404         _precision = precision if precision is not None else "f16" if mixed_precision else None
405         if _precision == "float16":
406             _precision = "f16"
407         elif _precision == "float8":
408             _precision = "f8"
409         elif _precision == "bfloat16":
410             _precision = "bf16"
411         accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps, mixed_precision=_precision)
412         if isinstance(model, nn.Module):
413             model = accelerator.prepare_model(model)
414 
415         optimizer = map_value_list_dict(optimizer, accelerator.prepare_optimizer)
416 
417         if isinstance(training_dataloader, torch.utils.data.DataLoader):
418             training_dataloader = accelerator.prepare_data_loader(training_dataloader)
419 
420         if scheduler is not None:
421             scheduler = map_value_list_dict(scheduler, accelerator.prepare_scheduler)
422 
423         return model, optimizer, training_dataloader, scheduler, accelerator
424 
425     def save_training_script(self) -> None:
426         """Save the training script to tracking dashboard and output path."""
427         file_path = Path(sys.argv[0])
428         if file_path.is_file():
429             file_name = file_path.name
430             self.dashboard_logger.add_artifact(file_or_dir=file_path, name=file_name, artifact_type="file")
431             with file_path.open(encoding="utf8") as f:
432                 self.dashboard_logger.add_text("training-script", f"{f.read()}", 0)
433             shutil.copyfile(file_path, self.output_path / file_name)
434 
435     @staticmethod
436     def init_loggers(
437         config: TrainerConfig,
438         output_path: str | os.PathLike[Any],
439         dashboard_logger: BaseDashboardLogger | None = None,
440         c_logger: ConsoleLogger | None = None,
441     ) -> tuple[BaseDashboardLogger, ConsoleLogger]:
442         """Init console and dashboard loggers.
443 
444         Use the given logger if passed externally else use config values to pick the right logger.
445         Return a dashboard logger only for the rank 0 process in DDP
446         Define a console logger for each process in DDP
447 
448         Args:
449             config (TrainerConfig): Model config.
450             output_path (str): Output path to save the training artifacts.
451             dashboard_logger (DashboardLogger): Object passed to the trainer from outside.
452             c_logger (ConsoleLogger): Object passed to the trained from outside.
453 
454         Returns:
455             Initialized dashboard_logger and console_logger objects.
456         """
457         c_logger = ConsoleLogger() if c_logger is None else c_logger
458 
459         # only allow dashboard logging for the main process in DDP mode
460         if get_rank() > 0:
461             return DummyLogger(), c_logger
462         if dashboard_logger is None:
463             dashboard_logger = logger_factory(config, output_path)
464         return dashboard_logger, c_logger
465 
466     def setup_small_run(self, small_run: int | None = None) -> None:
467         """Use a subset of samples for training, evaluation and testing."""
468         if small_run is not None:
469             logger.info("[!] Small Run, only using %i samples.", small_run)
470             self.train_samples = None if self.train_samples is None else self.train_samples[:small_run]
471             self.eval_samples = None if self.eval_samples is None else self.eval_samples[:small_run]
472             self.test_samples = None if self.test_samples is None else self.test_samples[:small_run]
473 
474     @staticmethod
475     def init_training(
476         args: TrainerArgs, coqpit_overrides: list[str], config: TrainerConfig | None = None
477     ) -> tuple[TrainerConfig, dict[str, str]]:
478         """Initialize training and update model configs from command line arguments.
479 
480         Args:
481             args: Parsed trainer arguments.
482             config_overrides: Parsed config overriding arguments.
483             config: Model config. If none, it is generated from `args`. Defaults to None.
484 
485         Returns:
486             config (TrainerConfig): Config paramaters.
487         """
488         # set arguments for continuing training
489         if args.continue_path:
490             config_path = os.path.join(args.continue_path, "config.json")
491             args.restore_path, best_model = get_last_checkpoint(args.continue_path)
492             if not args.best_path:
493                 args.best_path = best_model
494             # use the same config
495             if config:
496                 config.load_json(config_path)
497             else:
498                 config = TrainerConfig()
499                 config.load_json(config_path)
500 
501         if config is None:
502             msg = "Config or continue_path containing Config not provided"
503             raise ValueError(msg)
504 
505         # override config values from command-line args
506         # TODO: Maybe it is better to do it outside
507         if len(coqpit_overrides) > 0:
508             config.parse_known_args(coqpit_overrides, relaxed_parser=True)
509 
510         # update the config.json fields and copy it to the output folder
511         new_fields = {}
512         if args.rank == 0:
513             if args.restore_path:
514                 new_fields["restore_path"] = args.restore_path
515             new_fields["github_branch"] = get_git_branch()
516         return config, new_fields
517 
518     @staticmethod
519     def setup_training_environment(args: TrainerArgs, config: TrainerConfig, gpu: int | None) -> tuple[bool, int]:
520         if platform.system() != "Windows":
521             # https://github.com/pytorch/pytorch/issues/973
522             import resource  # pylint: disable=import-outside-toplevel
523 
524             rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
525             resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
526 
527         # set and initialize Pytorch runtime
528         use_cuda, num_gpus = setup_torch_training_env(
529             args=args,
530             cudnn_enable=config.cudnn_enable,
531             cudnn_deterministic=config.cudnn_deterministic,
532             cudnn_benchmark=config.cudnn_benchmark,
533             use_ddp=args.use_ddp,
534             training_seed=config.training_seed,
535             allow_tf32=config.allow_tf32,
536             gpu=gpu if args.gpu is None else args.gpu,
537         )
538 
539         print_training_env(args, config)
540         return use_cuda, num_gpus
541 
542     @staticmethod
543     @overload
544     def run_get_model(config: TrainerConfig, get_model: Callable[[TrainerConfig], TrainerModel]) -> TrainerModel: ...
545 
546     @staticmethod
547     @overload
548     def run_get_model(config: TrainerConfig, get_model: Callable[[], TrainerModel]) -> TrainerModel: ...
549 
550     @staticmethod
551     def run_get_model(config: TrainerConfig, get_model: Callable[..., TrainerModel]) -> TrainerModel:
552         """Run the `get_model` function and return the model.
553 
554         Args:
555             config (TrainerConfig): Model config.
556 
557         Returns:
558             TrainerModel: initialized model.
559         """
560         return get_model(config) if len(signature(get_model).parameters) == 1 else get_model()
561 
562     @staticmethod
563     def run_get_data_samples(
564         config: TrainerConfig, get_data_samples: Callable[..., list[Any]]
565     ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
566         if callable(get_data_samples):
567             if len(signature(get_data_samples).parameters) == 1:
568                 train_samples, eval_samples, test_samples = get_data_samples(config)
569             else:
570                 train_samples, eval_samples, test_samples = get_data_samples()
571             return train_samples, eval_samples, test_samples
572         return None, None, None
573 
574     def restore_model(self) -> None:
575         """Restore training from an old run.
576 
577         It restores model, optimizer, AMP scaler and training stats.
578         """
579 
580         def _restore_list_objs(states: Any, obj: Any) -> None:
581             if isinstance(obj, list):
582                 for idx, state in enumerate(states):
583                     obj[idx].load_state_dict(state)
584             elif isinstance(obj, dict):
585                 for key, state in states.items():
586                     obj[key].load_state_dict(state)
587             else:
588                 obj.load_state_dict(states)
589 
590         verb = "Continuing" if self.continue_run else "Restoring"
591         logger.info(" > %s from %s ...", verb, os.path.basename(self.args.restore_path))
592         checkpoint = load_fsspec(self.args.restore_path, map_location="cpu")
593 
594         try:
595             logger.info(" > Restoring Model...")
596             self.model.load_state_dict(checkpoint["model"])
597             if self.continue_run:
598                 logger.info(" > Restoring Optimizer...")
599                 try:
600                     _restore_list_objs(checkpoint["optimizer"], self.optimizer)
601                 except (KeyError, TypeError, RuntimeError):
602                     logger.info(" > Optimizer is not compatible with the restored model.")
603                 if checkpoint.get("scheduler"):
604                     logger.info(" > Restoring Scheduler...")
605                     _restore_list_objs(checkpoint["scheduler"], self.scheduler)
606                 if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
607                     logger.info(" > Restoring Scaler...")
608                     _restore_list_objs(checkpoint["scaler"], self.scaler)
609         except (KeyError, RuntimeError, ValueError):
610             logger.info(" > Partial model initialization...")
611             model_dict = self.model.state_dict()
612             model_dict = set_partial_state_dict(model_dict, checkpoint["model"], self.config)
613             self.model.load_state_dict(model_dict)
614             del model_dict
615 
616         self.total_steps_done = checkpoint["step"] + 1  # +1 not to immediately checkpoint if the model is restored
617         self.epochs_done = checkpoint["epoch"]
618 
619         if not self.continue_run:
620             self.total_steps_done = 0
621             self.epochs_done = 0
622             # Use LR read from the checkpoint if we continue a training run
623             self.reset_lr()
624 
625         logger.info(" > Model restored from step %i", checkpoint["step"])
626         torch.cuda.empty_cache()
627 
628     def reset_lr(self) -> None:
629         """Reset learning rate to default values."""
630         for key, optim in iter_value_list_dict(self.optimizer):
631             for group in optim.param_groups:
632                 lr = self.get_lr(self.model, self.config)
633                 group["lr"] = lr[key] if key is not None else lr  # type: ignore[index]
634 
635     #########################
636     # DATA LOADING FUNCTIONS
637     #########################
638 
639     def _get_loader(
640         self,
641         model: TrainerModel,
642         config: TrainerConfig,
643         assets: dict[str, Any],
644         samples: list[Any] | None,
645         *,
646         is_eval: bool,
647         verbose: bool,
648         num_gpus: int,
649     ) -> DataLoader[Any]:
650         loader = model.get_data_loader(
651             config=config,
652             assets=assets,
653             is_eval=is_eval,
654             samples=samples,
655             verbose=verbose,
656             num_gpus=num_gpus,
657             rank=self.args.rank,
658         )
659 
660         assert len(loader) > 0, (
661             " ❗ len(DataLoader) returns 0. Make sure your dataset is not empty or len(dataset) > 0. "
662         )
663         return loader
664 
665     def _get_model(self) -> TrainerModel:
666         if not hasattr(self, "wrapped_model") or self.wrapped_model is None:
667             return self.model
668         return self.wrapped_model
669 
670     def get_train_dataloader(
671         self, training_assets: dict[str, Any], samples: list[Any] | None, *, verbose: bool = True
672     ) -> DataLoader[Any]:
673         """Initialize and return a training data loader.
674 
675         Call ```model.get_train_data_loader``` if it is implemented, else call ```model.get_data_loader```
676         and set ```is_eval=False```.
677 
678         Args:
679             ap (AudioProcessor): Audio processor.
680             samples (List): Data samples used for training.
681             verbose (bool): enable/disable printing loader stats at initialization.
682 
683         Returns:
684             DataLoader: Initialized training data loader.
685         """
686         model = self._get_model()
687         try:
688             return model.get_train_data_loader(
689                 self.config,
690                 self.training_assets,
691                 samples,
692                 verbose,
693                 self.num_gpus,
694                 self.args.rank,
695             )
696         except NotImplementedError:
697             return self._get_loader(
698                 model,
699                 self.config,
700                 training_assets,
701                 samples,
702                 is_eval=False,
703                 verbose=verbose,
704                 num_gpus=self.num_gpus,
705             )
706 
707     def get_eval_dataloader(
708         self, training_assets: dict[str, Any], samples: list[Any] | None, *, verbose: bool
709     ) -> DataLoader[Any]:
710         """Initialize and return a evaluation data loader.
711 
712         Call ```model.get_eval_data_loader``` if it is implemented, else call ```model.get_data_loader```
713         and set ```is_eval=True```.
714 
715         Args:
716             ap (AudioProcessor): Audio processor.
717             samples (List): Data samples used for training.
718             verbose (bool): enable/disable printing loader stats at initialization.
719 
720         Returns:
721             DataLoader: Initialized training data loader.
722         """
723         model = self._get_model()
724         try:
725             return model.get_eval_data_loader(
726                 self.config,
727                 self.training_assets,
728                 samples,
729                 verbose,
730                 self.num_gpus,
731                 self.args.rank,
732             )
733         except NotImplementedError:
734             return self._get_loader(
735                 model,
736                 self.config,
737                 training_assets,
738                 samples,
739                 is_eval=True,
740                 verbose=verbose,
741                 num_gpus=self.num_gpus,
742             )
743 
744     def get_test_dataloader(
745         self, training_assets: dict[str, Any], samples: list[Any] | None, *, verbose: bool
746     ) -> DataLoader[Any]:
747         """Initialize and return a evaluation data loader.
748 
749         Call ```model.get_test_data_loader``` if it is implemented, else call ```model.get_data_loader```
750         and set ```is_eval=True```.
751 
752         Args:
753             ap (AudioProcessor): Audio processor.
754             samples (List): Data samples used for training.
755             verbose (bool): enable/disable printing loader stats at initialization.
756 
757         Returns:
758             DataLoader: Initialized training data loader.
759         """
760         model = self._get_model()
761         try:
762             return model.get_test_data_loader(
763                 self.config,
764                 self.training_assets,
765                 samples,
766                 verbose,
767                 self.num_gpus,
768                 self.args.rank,
769             )
770         except NotImplementedError:
771             return self._get_loader(
772                 model,
773                 self.config,
774                 training_assets,
775                 samples,
776                 is_eval=True,
777                 verbose=verbose,
778                 num_gpus=self.num_gpus,
779             )
780 
781     def format_batch(self, batch: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
782         """Format the dataloader output and return a batch.
783 
784         1. Call ```model.format_batch```.
785         2. Pass the batch to the Device.
786         3. Call ```model.format_batch_on_device```.
787 
788         Args:
789             batch (List): Batch returned by the dataloader.
790 
791         Returns:
792             Dict: Formatted batch.
793         """
794         with suppress(NotImplementedError):
795             batch = (
796                 self.wrapped_model.format_batch(batch)
797                 if self.wrapped_model is not None
798                 else self.model.format_batch(batch)
799             )
800 
801         if isinstance(batch, dict):
802             for k, v in batch.items():
803                 batch[k] = to_cuda(v)
804         elif isinstance(batch, list):
805             batch = [to_cuda(v) for v in batch]
806 
807         with suppress(NotImplementedError):
808             batch = (
809                 self.wrapped_model.format_batch_on_device(batch)
810                 if self.wrapped_model is not None
811                 else self.model.format_batch_on_device(batch)
812             )
813         return batch
814 
815     ######################
816     # TRAIN FUNCTIONS
817     ######################
818 
819     @staticmethod
820     def master_params(optimizer: torch.optim.Optimizer) -> Generator[Any]:
821         """Generator over parameters owned by the optimizer.
822 
823         Used to select parameters used by the optimizer for gradient clipping.
824 
825         Args:
826             optimizer: Target optimizer.
827         """
828         for group in optimizer.param_groups:
829             yield from group["params"]
830 
831     def _model_train_step(
832         self,
833         batch: dict[str, Any] | list[Any],
834         criterion: nn.Module | list[nn.Module],
835         optimizer_idx: int | None = None,
836     ) -> tuple[dict[str, Any], dict[str, Any]]:
837         """Perform a training forward step. Compute model outputs and losses.
838 
839         Args:
840             batch (Dict): [description]
841             criterion (nn.Module): [description]
842             optimizer_idx (int, optional): [description]. Defaults to None.
843 
844         Returns:
845             Tuple[Dict, Dict]: Model outputs and losses
846         """
847         input_args: list[Any] = [batch, criterion]
848         if optimizer_idx is not None:
849             input_args.append(optimizer_idx)
850         # unwrap model in DDP training
851         if self.wrapped_model is not None:
852             return self.wrapped_model.train_step(*input_args)
853         return self.model.train_step(*input_args)
854 
855     def _get_autocast_args(self, *, mixed_precision: bool, precision: str) -> tuple[str, torch.dtype]:
856         device = "cpu"
857         dtype = torch.get_autocast_dtype("cpu") if is_pytorch_at_least_2_4() else torch.get_autocast_cpu_dtype()
858         if self.use_cuda:
859             device = "cuda"
860             dtype = torch.float32
861             if mixed_precision:
862                 if precision == "fp16":
863                     dtype = torch.float16
864                 elif precision == "bf16":
865                     dtype = torch.bfloat16
866                 else:
867                     msg = f" ❗ Unknown precision {precision}"
868                     raise ValueError(msg)
869         elif mixed_precision:
870             dtype = torch.bfloat16
871         return device, dtype
872 
873     def detach_loss_dict(
874         self,
875         loss_dict: dict[str, Any],
876         *,
877         step_optimizer: bool,
878         optimizer_idx: int | None = None,
879         grad_norm: torch.Tensor | float | None = None,
880     ) -> dict[str, Any]:
881         # detach losses for logging
882         loss_dict_detached = self._detach_loss_dict(loss_dict)
883         # loss_dict_detached["loss"] = loss_di`ct_detached["loss"] * float(self.grad_accum_steps)
884 
885         if optimizer_idx is not None:
886             loss_dict_detached[f"loss_{optimizer_idx}"] = loss_dict_detached.pop("loss")
887             if step_optimizer and grad_norm is not None:
888                 loss_dict_detached[f"grad_norm_{optimizer_idx}"] = grad_norm
889         elif step_optimizer and grad_norm is not None:
890             loss_dict_detached["grad_norm"] = grad_norm
891         return loss_dict_detached
892 
893     def _compute_loss(
894         self,
895         batch: dict[str, Any] | list[Any],
896         criterion: nn.Module | list[nn.Module],
897         optimizer_idx: int | None,
898     ) -> tuple[dict[str, Any], dict[str, Any]]:
899         device, dtype = self._get_autocast_args(
900             mixed_precision=self.config.mixed_precision, precision=self.config.precision
901         )
902         with torch.autocast(device_type=device, dtype=dtype, enabled=self.config.mixed_precision):
903             if optimizer_idx is not None:
904                 outputs, loss_dict = self._model_train_step(batch, criterion, optimizer_idx=optimizer_idx)
905             else:
906                 outputs, loss_dict = self._model_train_step(batch, criterion)
907         return outputs, loss_dict
908 
909     @staticmethod
910     def _set_grad_clip_per_optimizer(config: TrainerConfig, optimizer_idx: int | None) -> float:
911         # set gradient clipping threshold
912         grad_clip: float = 0.0  # meaning no gradient clipping
913         if "grad_clip" in config and config.grad_clip is not None:
914             if optimizer_idx is not None:
915                 if isinstance(config.grad_clip, list):
916                     grad_clip = config.grad_clip[optimizer_idx]
917                 else:
918                     logger.warning(" [!] You are using multiple optimizers but `grad_clip` is not a list.")
919             else:
920                 if isinstance(config.grad_clip, list):
921                     msg = "`grad_clip` is a list, but no optimizer_idx specified"
922                     raise ValueError(msg)
923                 grad_clip = config.grad_clip
924         return grad_clip
925 
926     def _compute_grad_norm(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
927         return torch.norm(torch.cat([param.grad.view(-1) for param in self.master_params(optimizer)], dim=0), p=2)
928 
929     def _grad_clipping(
930         self, grad_clip: float, optimizer: torch.optim.Optimizer, scaler: Optional["torch.GradScaler"]
931     ) -> torch.Tensor:
932         """Perform gradient clipping."""
933         if grad_clip is not None and grad_clip > 0:
934             if scaler:
935                 scaler.unscale_(optimizer)
936             self.callbacks.before_gradient_clipping(self)
937             grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
938         else:
939             grad_norm = self._compute_grad_norm(optimizer)
940         return grad_norm
941 
942     def optimize(
943         self,
944         batch: dict[str, Any] | list[Any],
945         optimizer: torch.optim.Optimizer,
946         scaler: "torch.GradScaler | None",
947         criterion: nn.Module | list[nn.Module],
948         scheduler: LRScheduler | None,
949         *,
950         optimizer_idx: int | None = None,
951         step_optimizer: bool = True,
952         num_optimizers: int = 1,
953     ) -> tuple[dict[str, Any], dict[str, Any], float]:
954         """Perform a forward - backward pass and run the optimizer.
955 
956         Args:
957             batch (Dict): Input batch. If
958             optimizer (Union[nn.optim.Optimizer, List]): Model's optimizer. If it is a list then, `optimizer_idx` must be defined to indicate the optimizer in use.
959             scaler (AMPScaler): AMP scaler.
960             criterion (nn.Module): Model's criterion.
961             scheduler (LRScheduler): LR scheduler used by the optimizer.
962             optimizer_idx (int, optional): Target optimizer being used. Defaults to None.
963             step_optimizer (bool, optional): Whether step the optimizer. If False, gradients are accumulated and
964                 model parameters are not updated. Defaults to True.
965             num_optimizers (int, optional): Number of optimizers. Defaults to 1.
966 
967         Raises:
968             RuntimeError: When the loss is NaN.
969 
970         Returns:
971             Tuple[Dict, Dict, int, torch.Tensor]: model outputs, losses, step time and gradient norm.
972         """
973         step_start_time = time.time()
974 
975         # forward pass and loss computation
976         outputs, loss_dict = self._compute_loss(batch=batch, criterion=criterion, optimizer_idx=optimizer_idx)
977 
978         # skip the rest if not outputs from the model
979         if not loss_dict:
980             step_time = time.time() - step_start_time
981             return outputs, {}, step_time
982 
983         grad_clip = self._set_grad_clip_per_optimizer(config=self.config, optimizer_idx=optimizer_idx)
984         # optimizer step
985         grad_norm: float | torch.Tensor = 0.0
986         update_lr_scheduler = True
987 
988         # callback
989         self.callbacks.before_backward_pass(self, loss_dict)
990 
991         # accumulated gradients adjustment
992         loss_dict["loss"] = loss_dict["loss"] / float(self.grad_accum_steps)
993 
994         if self.use_accelerate:
995             with self.accelerator.accumulate(self.model):
996                 ctx_mgr = self.accelerator.autocast if self.config.mixed_precision else nullcontext
997                 with ctx_mgr():
998                     self.accelerator.backward(loss_dict["loss"])
999                     grad_norm = self._compute_grad_norm(optimizer)
1000                     if self.accelerator.sync_gradients and grad_clip is not None and grad_clip > 0:
1001                         self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
1002                     optimizer.step()
1003                     self._stepped_optimizers.add(optimizer_idx)
1004                     if (
1005                         scheduler is not None
1006                         and not self.config.scheduler_after_epoch
1007                         and not self.accelerator.optimizer_step_was_skipped
1008                     ):
1009                         scheduler.step()
1010                     optimizer.zero_grad(set_to_none=True)
1011         else:
1012             if self.use_amp_scaler and scaler is not None:
1013                 # model optimizer step in mixed precision mode
1014                 scaler.scale(loss_dict["loss"]).backward()
1015                 # gradient accumulation
1016                 if step_optimizer:
1017                     grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=scaler)
1018                     scale_prev = scaler.get_scale()
1019                     scaler.step(optimizer)
1020                     # update the scaler at the end of all the optimizer steps
1021                     if optimizer_idx is None or (optimizer_idx + 1 == num_optimizers):
1022                         scaler.update()
1023                         loss_dict["amp_scaler"] = scaler.get_scale()  # for logging
1024                     update_lr_scheduler = scale_prev <= scaler.get_scale()
1025             else:
1026                 # main model optimizer step
1027                 loss_dict["loss"].backward()
1028                 # gradient accumulation
1029                 if step_optimizer:
1030                     self.callbacks.before_gradient_clipping(self)
1031                     if grad_clip > 0:
1032                         grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
1033                     optimizer.step()
1034                     self._stepped_optimizers.add(optimizer_idx)
1035 
1036             # setup lr
1037             if (
1038                 scheduler is not None
1039                 and update_lr_scheduler
1040                 and not self.config.scheduler_after_epoch
1041                 and step_optimizer
1042             ):
1043                 scheduler.step()
1044 
1045             # zero-out optimizer
1046             if step_optimizer:
1047                 optimizer.zero_grad(set_to_none=True)
1048 
1049         # pytorch skips the step when the norm is 0. So ignore the norm value when it is NaN
1050         if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
1051             grad_norm = 0
1052 
1053         step_time = time.time() - step_start_time
1054 
1055         # detach loss dict
1056         loss_dict_detached = self.detach_loss_dict(
1057             loss_dict, step_optimizer=step_optimizer, optimizer_idx=optimizer_idx, grad_norm=grad_norm
1058         )
1059         return outputs, loss_dict_detached, step_time
1060 
1061     def train_step(
1062         self, batch: dict[str, Any] | list[Any], batch_n_steps: int, step: int, loader_start_time: float
1063     ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, dict[str, Any] | None]:
1064         """Perform a training step on a batch of inputs and log the process.
1065 
1066         Args:
1067             batch (Dict): Input batch.
1068             batch_n_steps (int): Number of steps needed to complete an epoch. Needed for logging.
1069             step (int): Current step number in this epoch.
1070             loader_start_time (float): The time when the data loading is started. Needed for logging.
1071 
1072         Returns:
1073             Tuple[Dict, Dict]: Model outputs and losses.
1074         """
1075         self.callbacks.on_train_step_start(self)
1076         # format data
1077         batch = self.format_batch(batch)
1078         loader_time = time.time() - loader_start_time
1079 
1080         # containers to hold model outputs and losses for each optimizer.
1081         outputs: dict[str, Any] | list[dict[str, Any]]
1082         loss_dict = {}
1083 
1084         # log learning rates (do it before they're updated in optimize())
1085         lrs = {}
1086         for key, optim in iter_value_list_dict(self.optimizer):
1087             name = f"current_lr_{key}" if key is not None else "current_lr"
1088             lrs[name] = optim.param_groups[0]["lr"]
1089         loss_dict.update(lrs)
1090 
1091         # OPTIMIZATION
1092         try:
1093             # custom optimize for the model
1094             step_time = time.time()
1095             device, dtype = self._get_autocast_args(
1096                 mixed_precision=self.config.mixed_precision, precision=self.config.precision
1097             )
1098             with torch.autocast(device_type=device, dtype=dtype, enabled=self.config.mixed_precision):
1099                 outputs, loss_dict_new = self.model.optimize(batch, self)
1100             step_time = time.time() - step_time
1101             # If None, skip the step
1102             if outputs is None:
1103                 return None, None
1104             # TODO: find a way to log grad_norm for custom optimize
1105             loss_dict_new = self.detach_loss_dict(loss_dict_new, step_optimizer=True)
1106             loss_dict.update(loss_dict_new)
1107         except NotImplementedError as e:
1108             # gradient accumulation
1109             # TODO: grad accumulation for each optimizer
1110             step_optimizer = True
1111             if ((step + 1) % self.grad_accum_steps != 0) and (step + 1 != batch_n_steps):
1112                 step_optimizer = False
1113 
1114             if not isinstance(self.optimizer, list):
1115                 if isinstance(self.scheduler, list):
1116                     msg = "Can't use list of schedulers with a single optimizer."
1117                     raise TypeError(msg) from e
1118                 if isinstance(self.optimizer, dict) or isinstance(self.scheduler, dict):
1119                     msg = "Can only use dict of optimizers/schedulers with custom `optimize()`"
1120                     raise TypeError(msg) from e
1121                 # auto training with a single optimizer
1122                 outputs, loss_dict_new, step_time = self.optimize(
1123                     batch,
1124                     self.optimizer,
1125                     self.scaler,
1126                     self.criterion,
1127                     self.scheduler,
1128                     step_optimizer=step_optimizer,
1129                     num_optimizers=1,
1130                 )
1131                 loss_dict.update(loss_dict_new)
1132             else:
1133                 if self.grad_accum_steps != 1:
1134                     msg = " [!] Coqui Trainer does not support grad_accum_steps for multiple-optimizer setup, please set grad_accum_steps to 1 or implement in your model a custom `optimize` method to deal with dangling gradients in multiple-optimizer setup!"
1135                     raise ValueError(msg) from e
1136                 # auto training with multiple optimizers (e.g. GAN)
1137                 outputs_per_optimizer = []
1138                 total_step_time = 0.0
1139                 for idx, optimizer in enumerate(self.optimizer):
1140                     criterion = self.criterion
1141                     # scaler = self.scaler[idx] if self.use_amp_scaler else None
1142                     scaler = self.scaler
1143                     scheduler = None
1144                     if self.scheduler is not None and isinstance(self.scheduler, list):
1145                         scheduler = self.scheduler[idx]
1146                     optimizer_outputs, loss_dict_new, step_time = self.optimize(
1147                         batch,
1148                         optimizer,
1149                         scaler,
1150                         criterion,
1151                         scheduler,
1152                         optimizer_idx=idx,
1153                         step_optimizer=step_optimizer,
1154                         num_optimizers=len(self.optimizer),
1155                     )
1156                     # skip the rest if the model returns None
1157                     total_step_time += step_time
1158                     outputs_per_optimizer.append(optimizer_outputs)
1159                     # merge loss_dicts from each optimizer
1160                     # rename duplicates with the optimizer idx
1161                     # if None, model skipped this optimizer
1162                     if loss_dict_new is not None:
1163                         for k, v in loss_dict_new.items():
1164                             if k in loss_dict:
1165                                 loss_dict[f"{k}-{idx}"] = v
1166                             else:
1167                                 loss_dict[k] = v
1168                     step_time = total_step_time
1169 
1170                 outputs = outputs_per_optimizer
1171 
1172                 # clear any pesky gradients after gradient accumulation
1173                 if step_optimizer:
1174                     self.model.zero_grad(set_to_none=True)
1175 
1176         if self.keep_avg_train is not None:
1177             # update avg runtime stats
1178             keep_avg_update = {}
1179             keep_avg_update["avg_loader_time"] = loader_time
1180             keep_avg_update["avg_step_time"] = step_time
1181             self.keep_avg_train.update_values(keep_avg_update)
1182 
1183             # update avg loss stats
1184             update_eval_values = {}
1185             for key, value in loss_dict.items():
1186                 update_eval_values["avg_" + key] = value
1187             self.keep_avg_train.update_values(update_eval_values)
1188 
1189         # print training progress
1190         if self.total_steps_done % self.config.print_step == 0:
1191             # log run-time stats
1192             loss_dict.update(
1193                 {
1194                     "step_time": round(step_time, 4),
1195                     "loader_time": round(loader_time, 4),
1196                 }
1197             )
1198             self.c_logger.print_train_step(
1199                 batch_n_steps,
1200                 step,
1201                 self.total_steps_done,
1202                 loss_dict,
1203                 self.keep_avg_train.avg_values if self.keep_avg_train is not None else {},
1204             )
1205 
1206         if self.args.rank == 0:
1207             # Plot Training Iter Stats
1208             # reduce TB load and don't log every step
1209             if self.total_steps_done % self.config.plot_step == 0:
1210                 self.dashboard_logger.train_step_stats(self.total_steps_done, loss_dict)
1211             if (
1212                 self.total_steps_done % self.config.save_step == 0
1213                 and self.total_steps_done != 0
1214                 and self.config.save_checkpoints
1215             ):
1216                 self.save_checkpoint()
1217 
1218             if self.total_steps_done % self.log_model_step == 0:
1219                 # log checkpoint as artifact
1220                 self.update_training_dashboard_logger(batch=batch, outputs=outputs)
1221 
1222             self.dashboard_logger.flush()
1223 
1224         self.total_steps_done += 1
1225         self.callbacks.on_train_step_end(self)
1226         return outputs, loss_dict
1227 
1228     def train_epoch(self) -> None:
1229         """Main entry point for the training loop. Run training on the all training samples."""
1230         # initialize the data loader
1231         if self.train_loader is None:
1232             self.train_loader = self.get_train_dataloader(
1233                 self.training_assets,
1234                 self.train_samples,
1235                 verbose=True,
1236             )
1237             self.train_loader = self.prepare_accelerate_loader(self.train_loader)
1238         # set model to training mode
1239         self.model.train()
1240         epoch_start_time = time.time()
1241 
1242         self.callbacks.on_train_epoch_start(self)
1243 
1244         self.c_logger.print_train_start()
1245         loader_start_time = time.time()
1246         # TRAINING EPOCH -> iterate over the training samples
1247         batch_num_steps = len(self.train_loader)
1248         for cur_step, batch in enumerate(self.train_loader):
1249             outputs, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
1250             if outputs is None:
1251                 logger.info(" [!] `train_step()` retuned `None` outputs. Skipping training step.")
1252                 continue
1253             del outputs
1254             loader_start_time = time.time()
1255 
1256             # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
1257             if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
1258                 self.eval_epoch()
1259                 self.model.train()
1260 
1261         epoch_time = time.time() - epoch_start_time
1262         self.callbacks.on_train_epoch_end(self)
1263 
1264         # scheduler step
1265         if self.scheduler is not None and self.config.scheduler_after_epoch:
1266             for idx, scheduler in iter_value_list_dict(self.scheduler):
1267                 if scheduler is not None and idx in self._stepped_optimizers:
1268                     scheduler.step()
1269         self._stepped_optimizers.clear()
1270         # plot self.epochs_done Stats
1271         if self.args.rank == 0:
1272             epoch_stats = {"epoch_time": epoch_time}
1273             if self.keep_avg_train is not None:
1274                 epoch_stats.update(self.keep_avg_train.avg_values)
1275             self.dashboard_logger.train_epoch_stats(self.total_steps_done, epoch_stats)
1276             if self.config.model_param_stats:
1277                 self.dashboard_logger.model_weights(self.model, self.total_steps_done)
1278         torch.cuda.empty_cache()
1279 
1280     #######################
1281     # EVAL FUNCTIONS
1282     #######################
1283 
1284     def _model_eval_step(
1285         self,
1286         batch: dict[str, Any],
1287         model: TrainerModel,
1288         criterion: nn.Module | list[nn.Module],
1289         optimizer_idx: int | None = None,
1290     ) -> tuple[dict[str, Any], dict[str, Any]]:
1291         """Perform a evaluation forward pass. Compute model outputs and losses with no gradients.
1292 
1293         Args:
1294             batch (Dict): IBatch of inputs.
1295             model (TrainerModel): Model to call evaluation.
1296             criterion (nn.Module): Model criterion.
1297             optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.
1298 
1299         Returns:
1300             Tuple[Dict, Dict]: model outputs and losses.
1301         """
1302         input_args: list[Any] = [batch, criterion]
1303         if optimizer_idx is not None:
1304             input_args.append(optimizer_idx)
1305 
1306         return self._get_model().eval_step(*input_args)
1307 
1308     def eval_step(
1309         self, batch: dict[str, Any] | list[Any], step: int
1310     ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, dict[str, Any] | None]:
1311         """Perform a evaluation step on a batch of inputs and log the process.
1312 
1313         Args:
1314             batch (Dict): Input batch.
1315             step (int): Current step number in this epoch.
1316 
1317         Returns:
1318             Tuple[Dict, Dict]: Model outputs and losses.
1319         """
1320         outputs: dict[str, Any] | list[dict[str, Any]]
1321         with torch.inference_mode():
1322             loss_dict: dict[str, Any] = {}
1323             model = self._get_model()
1324             if not isinstance(self.optimizer, list) or len(signature(model.eval_step).parameters) == 2:  # noqa: PLR2004
1325                 outputs, loss_dict = model.eval_step(batch, self.criterion)
1326                 if outputs is None:
1327                     return None, None
1328             else:
1329                 optimizer_outputs = []
1330                 for idx, _ in enumerate(self.optimizer):
1331                     outputs_, loss_dict_new = model.eval_step(batch, self.criterion, idx)
1332                     if outputs_ is None:
1333                         return None, None
1334                     optimizer_outputs.append(outputs_)
1335 
1336                     if loss_dict_new:
1337                         loss_dict_new[f"loss_{idx}"] = loss_dict_new.pop("loss")
1338                         loss_dict.update(loss_dict_new)
1339                 outputs = optimizer_outputs
1340 
1341             loss_dict = self._detach_loss_dict(loss_dict)
1342 
1343             # update avg stats
1344             if self.keep_avg_eval is not None:
1345                 update_eval_values = {}
1346                 for key, value in loss_dict.items():
1347                     update_eval_values["avg_" + key] = value
1348                 self.keep_avg_eval.update_values(update_eval_values)
1349 
1350             if self.config.print_eval:
1351                 self.c_logger.print_eval_step(
1352                     step, loss_dict, self.keep_avg_eval.avg_values if self.keep_avg_eval is not None else {}
1353                 )
1354 
1355         return outputs, loss_dict
1356 
1357     @torch.inference_mode()
1358     def eval_epoch(self) -> None:
1359         """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""
1360         # initialize it when eval_epoch is called alone.
1361         self.keep_avg_eval = KeepAverage() if self.keep_avg_eval is None else self.keep_avg_eval
1362 
1363         if self.eval_loader is None:
1364             self.eval_loader = (
1365                 self.get_eval_dataloader(
1366                     self.training_assets,
1367                     self.eval_samples,
1368                     verbose=True,
1369                 )
1370                 if self.config.run_eval
1371                 else None
1372             )
1373 
1374         self.model.eval()
1375         self.c_logger.print_eval_start()
1376         loader_start_time = time.time()
1377         batch = None
1378         outputs = None
1379         for cur_step, batch in enumerate(self.eval_loader):  # type: ignore[arg-type]
1380             # format data
1381             batch = self.format_batch(batch)
1382             loader_time = time.time() - loader_start_time
1383             self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
1384             outputs_, _ = self.eval_step(batch, cur_step)
1385             if outputs_ is None:
1386                 logger.info(" [!] `eval_step()` retuned `None` outputs. Skipping evaluation step.")
1387                 continue
1388             outputs = outputs_
1389             loader_start_time = time.time()
1390         # plot epoch stats, artifacts and figures
1391         if self.args.rank == 0 and outputs is not None:
1392             model = self._get_model()
1393             with suppress(NotImplementedError):
1394                 model.eval_log(
1395                     batch,
1396                     outputs,
1397                     self.dashboard_logger,
1398                     self.training_assets,
1399                     self.total_steps_done,
1400                 )
1401             self.dashboard_logger.eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)
1402         torch.cuda.empty_cache()
1403 
1404     ##################################
1405     # TESTING
1406     ##################################
1407     def test_run(self) -> None:
1408         """Run model test.
1409 
1410         Test run is expected to pass over test samples and produce logging artifacts.
1411 
1412         If ```model.test_run()``` is defined, it will be called and it is expected to set and execute everything
1413         in the model.
1414 
1415         Else if  ```mode.test()``` is defined, it will be called and it takes an test data loader as an argument
1416         and iterate over it.
1417         """
1418         self.model.eval()
1419         model = self._get_model()
1420         test_outputs = None
1421         try:
1422             test_outputs = model.test_run(self.training_assets)
1423         except NotImplementedError:
1424             self.test_loader = self.get_test_dataloader(
1425                 self.training_assets,
1426                 self.test_samples if self.test_samples else self.eval_samples,
1427                 verbose=True,
1428             )
1429             # use test_loader to load test samples
1430             with suppress(NotImplementedError):
1431                 test_outputs = model.test(self.training_assets, self.test_loader, None)
1432         with suppress(NotImplementedError):
1433             model.test_log(test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done)
1434 
1435     def _restore_best_loss(self) -> None:
1436         """Restore the best loss.
1437 
1438         Restore from the args.best_path if provided else from the model
1439         (`args.continue_path`) used for resuming the training.
1440         """
1441         if self.continue_run and (self.total_steps_done != 0 or self.args.best_path):
1442             logger.info(" > Restoring best loss from %s ...", os.path.basename(self.args.best_path))
1443             ch = load_fsspec(self.args.restore_path, map_location="cpu")
1444             if "model_loss" in ch:
1445                 if isinstance(ch["model_loss"], dict):
1446                     self.best_loss = cast(LossDict, ch["model_loss"])
1447                 # For backwards-compatibility:
1448                 elif isinstance(ch["model_loss"], float):
1449                     if self.config.run_eval:
1450                         self.best_loss = {"train_loss": float("inf"), "eval_loss": ch["model_loss"]}
1451                     else:
1452                         self.best_loss = {"train_loss": ch["model_loss"], "eval_loss": None}
1453             logger.info(" > Starting with loaded last best loss %s", self.best_loss)
1454 
1455     def test(self, model: TrainerModel | None = None, test_samples: list[str] | None = None) -> None:
1456         """Run evaluation steps on the test data split.
1457 
1458         You can either provide the model and the test samples
1459         explicitly or the trainer uses values from the initialization.
1460 
1461         Args:
1462             model (TrainerModel, optional): Model to use for testing. If None, use the model given in the initialization.
1463                 Defaults to None.
1464 
1465             test_samples (List[str], optional): List of test samples to use for testing. If None, use the test samples
1466                 given in the initialization. Defaults to None.
1467         """
1468         logger.info(" > USING TEST SET...")
1469         self.keep_avg_eval = KeepAverage()
1470 
1471         if model is not None:
1472             self.model = model
1473 
1474         eval_samples_cache = self.eval_samples
1475         if test_samples is not None:
1476             self.eval_samples = test_samples
1477         else:
1478             self.eval_samples = self.test_samples
1479 
1480         self.eval_epoch()
1481         self.c_logger.print_epoch_end(self.epochs_done, self.keep_avg_eval.avg_values)
1482         self.eval_samples = eval_samples_cache
1483 
1484     ###################################
1485     # FIT FUNCTIONS
1486     ###################################
1487 
1488     def _fit(self) -> None:
1489         """🏃 train -> evaluate -> test for the number of epochs."""
1490         self._restore_best_loss()
1491 
1492         for epoch in range(self.epochs_done, self.config.epochs):
1493             if self.num_gpus > 1:
1494                 # let all processes sync up before starting with a new epoch of training
1495                 dist.barrier()
1496             self.callbacks.on_epoch_start(self)
1497             self.keep_avg_train = KeepAverage()
1498             self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
1499             self.epochs_done = epoch
1500             self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
1501             if not self.skip_train_epoch and not self.start_with_eval:
1502                 self.train_epoch()
1503             if self.config.run_eval:
1504                 self.eval_epoch()
1505             if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
1506                 self.test_run()
1507 
1508             self.c_logger.print_epoch_end(
1509                 epoch,
1510                 self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,  # type: ignore[union-attr]
1511             )
1512             if self.args.rank in [None, 0]:
1513                 self.save_best_model()
1514             self.callbacks.on_epoch_end(self)
1515             self.start_with_eval = False
1516 
1517     def fit_with_largest_batch_size(self, starting_batch_size: int = 2048) -> None:
1518         cuda_meminfo()
1519         bs = starting_batch_size
1520         while True:
1521             gc.collect()
1522             torch.cuda.empty_cache()
1523             try:
1524                 gc.collect()
1525                 torch.cuda.empty_cache()
1526                 self.config.batch_size = bs
1527                 logger.info(" > current batch size: %i", self.config.batch_size)
1528                 self._fit()
1529             except RuntimeError as exception:
1530                 if bs > 1 and should_reduce_batch_size(exception):
1531                     bs //= 2
1532                     gc.collect()
1533                     torch.cuda.empty_cache()
1534                 else:
1535                     raise
1536             except Exception as exception:  # pylint: disable=broad-except
1537                 # catches the torch.cuda.OutOfMemoryError
1538                 if bs > 1 and should_reduce_batch_size(exception):
1539                     bs //= 2
1540                     gc.collect()
1541                     torch.cuda.empty_cache()
1542                 else:
1543                     raise
1544             else:
1545                 break
1546 
1547     def fit(self) -> None:
1548         """Where the ✨️magic✨️ happens..."""
1549         try:
1550             self._fit()
1551             if self.args.rank == 0:
1552                 self.dashboard_logger.finish()
1553         except KeyboardInterrupt:
1554             logger.info(" > Keyboard interrupt detected.")
1555             if self.config.save_on_interrupt:
1556                 logger.info(" > Saving model before exiting...")
1557                 # save the model on keyboard interrupt
1558                 self.save_checkpoint()
1559                 # update the training dashboard logger
1560                 self.update_training_dashboard_logger()
1561             # call the keyboard interrupt callback
1562             self.callbacks.on_keyboard_interrupt(self)
1563             # if the output folder is empty remove the run.
1564             remove_experiment_folder(self.output_path)
1565             # clear the DDP processes
1566             if self.num_gpus > 1:
1567                 dist.destroy_process_group()
1568             # finish the wandb run and sync data
1569             if self.args.rank == 0:
1570                 self.dashboard_logger.finish()
1571             # stop without error signal
1572             try:
1573                 sys.exit(130)
1574             except SystemExit:
1575                 os._exit(130)  # pylint: disable=protected-access
1576         except BaseException:  # pylint: disable=broad-except
1577             remove_experiment_folder(self.output_path)
1578             traceback.print_exc()
1579             sys.exit(1)
1580 
1581     def profile_fit(
1582         self, torch_profiler: torch.profiler.profile, epochs: int | None = None, small_run: int | None = None
1583     ) -> torch.profiler.profile:
1584         """Run training under the torch profiler.
1585 
1586         Example::
1587             Run torch profiler to profile CPU, GPU and memory usage with Tensorboard logging.
1588 
1589             >>> import torch
1590             >>> profiler = torch.profiler.profile(
1591             >>>        activities=[
1592             >>>         torch.profiler.ProfilerActivity.CPU,
1593             >>>         torch.profiler.ProfilerActivity.CUDA,
1594             >>>     ],
1595             >>>     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
1596             >>>     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
1597             >>>     record_shapes=True,
1598             >>>     profile_memory=True,
1599             >>>     with_stack=True,
1600             >>> )
1601             >>> prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
1602         """
1603         self.dashboard_logger = DummyLogger()
1604         # train the model for a custom number of epochs
1605         if epochs:
1606             self.config.epochs = epochs
1607         # use a smaller set of training samples for profiling
1608         if small_run:
1609             self.setup_small_run(small_run)
1610         # run profiler
1611         self.config.run_eval = False
1612         self.config.test_delay_epochs = 9999999
1613         # set a callback to progress the profiler
1614         self.callbacks_on_train_step_end = [  # pylint: disable=attribute-defined-outside-init
1615             lambda trainer: trainer.torch_profiler.step()
1616         ]
1617         # set the profiler to access in the Trainer
1618         self.torch_profiler = torch_profiler  # pylint: disable=attribute-defined-outside-init
1619         # set logger output for Tensorboard
1620         # self.torch_profiler.on_trace_ready = torch.profiler.tensorboard_trace_handler(self.output_path)
1621         self.torch_profiler.start()
1622         self.fit()
1623         self.torch_profiler.stop()
1624         return self.torch_profiler
1625 
1626     @rank_zero_only
1627     def save_best_model(self) -> None:
1628         """Save the best model. It only saves if the current target loss is smaller then the previous."""
1629         eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
1630         train_loss = self._pick_target_avg_loss(self.keep_avg_train) or float("inf")
1631 
1632         # save the model and update the best_loss
1633         self.best_loss = save_best_model(
1634             {"train_loss": train_loss, "eval_loss": eval_loss},
1635             self.best_loss,
1636             self.config,
1637             self._get_model(),
1638             self.output_path,
1639             current_step=self.total_steps_done,
1640             epoch=self.epochs_done,
1641             optimizer=self.optimizer,
1642             scheduler=self.scheduler,
1643             scaler=self.scaler if self.use_amp_scaler else None,
1644             keep_all_best=self.config.save_all_best,
1645             keep_after=self.config.save_best_after,
1646             save_func=self.dashboard_logger.save_model,
1647         )
1648 
1649     @rank_zero_only
1650     def save_checkpoint(self) -> None:
1651         """Save the current model checkpoint."""
1652         eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
1653         train_loss = self._pick_target_avg_loss(self.keep_avg_train)
1654 
1655         save_checkpoint(
1656             self.config,
1657             self._get_model(),
1658             self.output_path,
1659             current_step=self.total_steps_done,
1660             epoch=self.epochs_done,
1661             optimizer=self.optimizer,
1662             scheduler=self.scheduler,
1663             scaler=self.scaler if self.use_amp_scaler else None,
1664             model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
1665             save_n_checkpoints=self.config.save_n_checkpoints,
1666             save_func=self.dashboard_logger.save_model,
1667         )
1668 
1669     @rank_zero_only
1670     def update_training_dashboard_logger(
1671         self, batch: dict[str, Any] | list[Any] | None = None, outputs: dict[str, Any] | None = None
1672     ) -> None:
1673         aliases = [
1674             f"epoch-{self.epochs_done}",
1675             f"step-{self.total_steps_done}",
1676         ]
1677         self.dashboard_logger.add_artifact(
1678             file_or_dir=self.output_path, name="checkpoint", artifact_type="model", aliases=aliases
1679         )
1680 
1681         # training visualizations
1682         if batch is not None and outputs is not None:
1683             model = self._get_model()
1684             with suppress(NotImplementedError):
1685                 model.train_log(
1686                     batch,
1687                     outputs,
1688                     self.dashboard_logger,
1689                     self.training_assets,
1690                     self.total_steps_done,
1691                 )
1692 
1693     #####################
1694     # GET FUNCTIONS
1695     #####################
1696 
1697     @staticmethod
1698     def get_optimizer(model: TrainerModel, config: TrainerConfig) -> ValueListDict[torch.optim.Optimizer]:
1699         """Return the optimizer.
1700 
1701         From the model if model implements `get_optimizer()` else
1702         check the optimizer parameters in the config and try initiating the optimizer.
1703 
1704         Args:
1705             model (TrainerModel): Training model.
1706             config (TrainerConfig): Training configuration.
1707 
1708         Returns:
1709             Union[torch.optim.Optimizer, List]: A optimizer or a list of optimizers. GAN models define a list.
1710         """
1711         try:
1712             return model.get_optimizer()
1713         except NotImplementedError as e:
1714             if isinstance(config.optimizer, list):
1715                 optimizers = []
1716                 for i, optimizer_name in enumerate(config.optimizer):
1717                     optimizer_params = {} if config.optimizer_params is None else config.optimizer_params[i]  # type: ignore[index]
1718                     optimizers.append(get_optimizer(optimizer_name, optimizer_params, config.lr, model))  # type: ignore[arg-type]
1719                 return optimizers
1720             if config.optimizer is None:
1721                 msg = "No name specified in `optimizer`"
1722                 raise ValueError(msg) from e
1723             optimizer_name = config.optimizer
1724             optimizer_params = {} if config.optimizer_params is None else config.optimizer_params
1725             return get_optimizer(optimizer_name, optimizer_params, config.lr, model)  # type: ignore[arg-type]
1726 
1727     @staticmethod
1728     def get_lr(model: TrainerModel, config: TrainerConfig) -> float | list[float] | dict[str, float]:
1729         """Set the initial learning rate.
1730 
1731         According to the model if model implements `get_lr()` else try setting
1732         the learning rate from the config.
1733 
1734         Args:
1735             model (TrainerModel): Training model.
1736             config (TrainerConfig): Training configuration.
1737 
1738         Returns:
1739             Union[float, List[float]]: A single learning rate or a list of learning rates, one for each optimzier.
1740         """
1741         try:
1742             return model.get_lr()
1743         except NotImplementedError:
1744             return config.lr
1745 
1746     @staticmethod
1747     def get_scheduler(
1748         model: TrainerModel,
1749         config: TrainerConfig,
1750         optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | dict[str, torch.optim.Optimizer],
1751     ) -> ValueListDict[LRScheduler] | None:
1752         """Return the scheduler.
1753 
1754         From the model if model implements `get_scheduler()` else
1755         check the config and try initiating the scheduler.
1756 
1757         Args:
1758             model (TrainerModel): Training model.
1759             config (TrainerConfig): Training configuration.
1760 
1761         Returns:
1762             Union[torch.optim.Optimizer, List, Dict]: A scheduler or a list of schedulers, one for each optimizer.
1763         """
1764         try:
1765             return model.get_scheduler(optimizer)
1766         except NotImplementedError:
1767             lr_scheduler = config.lr_scheduler
1768             lr_scheduler_params = config.lr_scheduler_params
1769             return get_scheduler(lr_scheduler, lr_scheduler_params, optimizer)  # type: ignore[arg-type]
1770 
1771     @staticmethod
1772     def get_criterion(model: TrainerModel) -> nn.Module | list[nn.Module]:
1773         """Receive the criterion from the model. Model must implement `get_criterion()`.
1774 
1775         Args:
1776             model (TrainerModel): Training model.
1777 
1778         Returns:
1779             nn.Module: Criterion layer.
1780         """
1781         return model.get_criterion()
1782 
1783     ####################
1784     # HELPER FUNCTIONS
1785     ####################
1786 
1787     @staticmethod
1788     def _detach_loss_dict(loss_dict: dict[str, Any]) -> dict[str, Any]:
1789         """Detach loss values from autograp.
1790 
1791         Args:
1792             loss_dict (Dict): losses.
1793 
1794         Returns:
1795             Dict: losses detached from autograph.
1796         """
1797         loss_dict_detached = {}
1798         for key, value in loss_dict.items():
1799             if isinstance(value, (int | float)):
1800                 loss_dict_detached[key] = value
1801             else:
1802                 loss_dict_detached[key] = value.detach().cpu().item()
1803         return loss_dict_detached
1804 
1805     def _pick_target_avg_loss(self, keep_avg_target: KeepAverage | None) -> float | None:
1806         """Pick the target loss to compare models."""
1807         # if the keep_avg_target is None or empty return None
1808         if keep_avg_target is None or len(list(keep_avg_target.avg_values.keys())) == 0:
1809             return None
1810 
1811         # return if target loss defined in the model config
1812         # if not available in Dict use loss_1 as by default loss
1813         if "target_loss" in self.config and self.config.target_loss:
1814             if f"avg_{self.config.target_loss}" in keep_avg_target.avg_values:
1815                 return keep_avg_target[f"avg_{self.config.target_loss}"]
1816 
1817             msg = " [!] Target loss not found in the keep_avg_target. You might be exiting the training loop before it is computed or set the target_loss in the model config incorrectly."
1818             raise ValueError(msg)
1819 
1820         # take the average of loss_{optimizer_idx} as the target loss when there are multiple optimizers
1821         if isinstance(self.optimizer, list):
1822             target_avg_loss = 0.0
1823             for idx in range(len(self.optimizer)):
1824                 if f"avg_loss_{idx}" in keep_avg_target.avg_values:
1825                     target_avg_loss += keep_avg_target[f"avg_loss_{idx}"]
1826             target_avg_loss /= len(self.optimizer)
1827         else:
1828             target_avg_loss = keep_avg_target.avg_values.get("avg_loss", 0)
1829         return target_avg_loss
1830 
1831     def _setup_logger_config(self, log_file: str) -> None:
1832         """Set up the logger based on the process rank in DDP."""
1833         logger_new = logging.getLogger("trainer")
1834         handler = logging.FileHandler(log_file, mode="a")
1835         fmt = logging.Formatter("")
1836         handler.setFormatter(fmt)
1837         logger_new.addHandler(handler)
1838 
1839         # only log to a file if rank > 0 in DDP
1840         if self.args.rank > 0:
1841             logger_new.handlers = [h for h in logger_new.handlers if not isinstance(h, logging.StreamHandler)]