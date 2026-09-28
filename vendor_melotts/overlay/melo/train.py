# flake8: noqa: E402

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from melo.download_utils import load_pretrain_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(
#     True
# )  # Not available if torch version is lower than 2.0
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


class SingleGPUWrapper(torch.nn.Module):
    """Compatibility wrapper exposing .module without DDP machinery."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)



def run():
    hps = utils.get_hparams()

    expected_pretrain_g = os.environ.get(
        "MELOTTS_EXPECT_PRETRAIN_G"
    )

    if expected_pretrain_g:
        expected_pretrain_g = os.path.realpath(
            expected_pretrain_g
        )

        actual_pretrain_g = getattr(
            hps,
            "pretrain_G",
            None,
        )

        actual_pretrain_g = (
            os.path.realpath(actual_pretrain_g)
            if actual_pretrain_g
            else None
        )

        if actual_pretrain_g != expected_pretrain_g:
            raise RuntimeError(
                "Explicit pretrained generator check failed: "
                f"expected={expected_pretrain_g!r}, "
                f"runtime={actual_pretrain_g!r}"
            )

        print(
            "Verified explicit generator pretrain:",
            actual_pretrain_g,
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    single_gpu = world_size == 1

    if single_gpu:
        # DDP provides no benefit with a single GPU and introduces
        # reducer/bucket/Gloo synchronization into every backward pass.
        rank = 0
        n_gpus = 1
        print("Single-GPU mode: DDP/Gloo disabled")
    else:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=local_rank,
        )
        rank = dist.get_rank()
        n_gpus = dist.get_world_size()
    
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    _train_workers = int(
        getattr(
            hps.train,
            "num_workers",
            0,
        )
    )

    _train_pin_memory = bool(
        getattr(
            hps.train,
            "pin_memory",
            False,
        )
    )

    _train_persistent_workers = bool(
        getattr(
            hps.train,
            "persistent_workers",
            False,
        )
    )

    if _train_workers < 0:
        raise RuntimeError(
            "train.num_workers must be >= 0"
        )

    if (
        _train_workers == 0
        and _train_persistent_workers
    ):
        raise RuntimeError(
            "persistent_workers=True requires "
            "num_workers > 0"
        )

    if _train_workers > 0:
        _train_prefetch_factor = int(
            getattr(
                hps.train,
                "prefetch_factor",
                2,
            )
        )

        if _train_prefetch_factor <= 0:
            raise RuntimeError(
                "train.prefetch_factor must be > 0"
            )
    else:
        _train_prefetch_factor = None

    if rank == 0:
        print(
            "Train DataLoader: "
            f"workers={_train_workers}, "
            f"prefetch_factor={_train_prefetch_factor}, "
            f"pin_memory={_train_pin_memory}, "
            "persistent_workers="
            f"{_train_persistent_workers}"
        )

    train_loader = DataLoader(
        train_dataset,
        num_workers=_train_workers,
        shuffle=False,
        pin_memory=_train_pin_memory,
        prefetch_factor=_train_prefetch_factor,
        persistent_workers=_train_persistent_workers,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)

    # ---------------------------------------------------------
    # Optional generator-freeze modes for controlled adaptation.
    #
    # full:
    #   historical MeloTTS behavior; entire generator trainable.
    #
    # conditioning_only:
    #   freeze the acoustic/phonetic trunk and train only the
    #   speaker anchor + speaker/style residual embeddings.
    #
    # Freeze MUST happen before optim_g is constructed because
    # optim_g intentionally contains only requires_grad params.
    # ---------------------------------------------------------

    generator_train_mode = os.environ.get(
        "MELOTTS_GENERATOR_TRAIN_MODE",
        "full",
    )

    if generator_train_mode == "full":
        pass

    elif generator_train_mode == "conditioning_only":
        expected_trainable = {
            "emb_g.weight",
            "emb_style.weight",
        }

        existing_names = {
            name
            for name, _ in net_g.named_parameters()
        }

        missing = (
            expected_trainable
            - existing_names
        )

        if missing:
            raise RuntimeError(
                "conditioning_only requested but "
                "required parameters are missing: "
                + repr(sorted(missing))
            )

        for name, param in net_g.named_parameters():
            param.requires_grad = (
                name in expected_trainable
            )

        actual_trainable = {
            name
            for name, param
            in net_g.named_parameters()
            if param.requires_grad
        }

        if actual_trainable != expected_trainable:
            raise RuntimeError(
                "Unexpected generator trainable set: "
                f"expected={sorted(expected_trainable)!r}, "
                f"actual={sorted(actual_trainable)!r}"
            )

    elif generator_train_mode == "new_style_only":
        # Train only the 12 D2 rows that did not exist in C.
        #
        # emb_style is one dense Parameter, therefore requires_grad
        # cannot freeze individual rows. Freeze every other generator
        # tensor and mask the emb_style gradient row-wise instead.
        new_style_rows = [
            6, 7, 8,
            14, 15, 16, 17,
            22, 23, 24, 25, 26,
        ]

        for name, param in net_g.named_parameters():
            param.requires_grad = (
                name == "emb_style.weight"
            )

        if tuple(
            net_g.emb_style.weight.shape
        ) != (27, 256):
            raise RuntimeError(
                "Unexpected emb_style shape for "
                "new_style_only: "
                + repr(
                    tuple(
                        net_g.emb_style.weight.shape
                    )
                )
            )

        grad_mask = torch.zeros_like(
            net_g.emb_style.weight
        )

        grad_mask[
            new_style_rows,
            :
        ] = 1.0

        # Keep the mask alive on the module and mask every gradient.
        net_g._new_style_grad_mask = grad_mask

        # Capture the mask itself in the hook closure.
        #
        # Do NOT reference net_g here: after optimizer creation,
        # single-GPU training replaces net_g with SingleGPUWrapper,
        # while this hook remains attached to the original
        # emb_style Parameter.
        _new_style_grad_mask = (
            net_g._new_style_grad_mask
        )

        net_g.emb_style.weight.register_hook(
            lambda grad, mask=_new_style_grad_mask:
                grad * mask
        )

        actual_trainable = {
            name
            for name, param
            in net_g.named_parameters()
            if param.requires_grad
        }

        if actual_trainable != {
            "emb_style.weight"
        }:
            raise RuntimeError(
                "Unexpected new_style_only "
                "trainable tensors: "
                + repr(
                    sorted(
                        actual_trainable
                    )
                )
            )

        if rank == 0:
            print(
                "new_style_only rows:",
                new_style_rows,
            )

            print(
                "new_style_only gradient-enabled "
                "elements:",
                len(new_style_rows)
                * net_g.emb_style.embedding_dim,
            )

    else:
        raise RuntimeError(
            "Unknown MELOTTS_GENERATOR_TRAIN_MODE: "
            + repr(generator_train_mode)
        )

    trainable_generator_params = sum(
        p.numel()
        for p in net_g.parameters()
        if p.requires_grad
    )

    total_generator_params = sum(
        p.numel()
        for p in net_g.parameters()
    )

    if rank == 0:
        print(
            "Generator train mode:",
            generator_train_mode,
        )

        print(
            "Generator trainable parameters:",
            trainable_generator_params,
            "/",
            total_generator_params,
        )

        print(
            "Generator trainable tensors:",
            [
                name
                for name, param
                in net_g.named_parameters()
                if param.requires_grad
            ],
        )

    if generator_train_mode == "conditioning_only":
        if trainable_generator_params != 7680:
            raise RuntimeError(
                "Expected exactly 7680 trainable generator "
                "parameters in conditioning_only mode, got "
                f"{trainable_generator_params}"
            )

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    generator_lr = float(
        os.environ.get(
            "MELOTTS_GENERATOR_LR",
            hps.train.learning_rate,
        )
    )

    if generator_train_mode == "new_style_only":
        generator_weight_decay = 0.0
    else:
        generator_weight_decay = float(
            os.environ.get(
                "MELOTTS_GENERATOR_WEIGHT_DECAY",
                "0.01",
            )
        )

    if rank == 0:
        print(
            "Generator optimizer LR:",
            generator_lr,
        )

        print(
            "Generator optimizer weight_decay:",
            generator_weight_decay,
        )

    optim_g = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            net_g.parameters(),
        ),
        generator_lr,
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay=generator_weight_decay,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    if single_gpu:
        net_g = SingleGPUWrapper(net_g)
        net_d = SingleGPUWrapper(net_d)
    else:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    
    # Explicit local pretrained paths always have priority.
    # For smoke tests we can intentionally train from scratch because
    # the upstream pretrained S3 files may be unavailable.
    if not any((hps.pretrain_G, hps.pretrain_D, hps.pretrain_dur)):
        if os.environ.get("MELOTTS_SKIP_PRETRAIN", "0") == "1":
            if rank == 0:
                print("MELOTTS_SKIP_PRETRAIN=1: starting from random initialization")
        else:
            try:
                pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()
                hps.pretrain_G = pretrain_G
                hps.pretrain_D = pretrain_D
                hps.pretrain_dur = pretrain_dur
            except Exception as e:
                if rank == 0:
                    print(f"WARNING: failed to download upstream pretrained models: {e}")
                    print("Falling back to random initialization.")

    if hps.pretrain_G:
        utils.load_checkpoint(
                hps.pretrain_G,
                net_g,
                None,
                skip_optimizer=True
            )
    if hps.pretrain_D:
        utils.load_checkpoint(
                hps.pretrain_D,
                net_d,
                None,
                skip_optimizer=True
            )


    if net_dur_disc is not None:
        if single_gpu:
            net_dur_disc = SingleGPUWrapper(net_dur_disc)
        else:
            net_dur_disc = DDP(
                net_dur_disc,
                device_ids=[rank],
                find_unused_parameters=True,
            )
        if hps.pretrain_dur:
            utils.load_checkpoint(
                    hps.pretrain_dur,
                    net_dur_disc,
                    None,
                    skip_optimizer=True
                )
                
    resume_batch_idx = 0

    try:
        if net_dur_disc is not None:
            dur_ckpt_path = utils.latest_checkpoint_path(
                hps.model_dir, "DUR_*.pth"
            )
            g_ckpt_path = utils.latest_checkpoint_path(
                hps.model_dir, "G_*.pth"
            )
            d_ckpt_path = utils.latest_checkpoint_path(
                hps.model_dir, "D_*.pth"
            )

            _, _, dur_resume_lr, dur_epoch = utils.load_checkpoint(
                dur_ckpt_path,
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_g, g_resume_lr, g_epoch = utils.load_checkpoint(
                g_ckpt_path,
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_d, d_resume_lr, d_epoch = utils.load_checkpoint(
                d_ckpt_path,
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )

            if not (dur_epoch == g_epoch == d_epoch):
                raise RuntimeError(
                    f"Checkpoint epoch mismatch: "
                    f"DUR={dur_epoch}, G={g_epoch}, D={d_epoch}"
                )

            epoch_str = g_epoch

            def _checkpoint_step(path, prefix):
                name = os.path.basename(path)
                return int(name[len(prefix):-4])

            dur_step = _checkpoint_step(dur_ckpt_path, "DUR_")
            g_step = _checkpoint_step(g_ckpt_path, "G_")
            d_step = _checkpoint_step(d_ckpt_path, "D_")

            if not (dur_step == g_step == d_step):
                raise RuntimeError(
                    f"Checkpoint step mismatch: "
                    f"DUR={dur_step}, G={g_step}, D={d_step}"
                )

            # Checkpoint is written after the current batch has completed.
            # Continue from the next global step and next batch.
            global_step = g_step + 1
            resume_batch_idx = (g_step % len(train_loader)) + 1

            print(
                f"Exact resume: epoch={epoch_str}, "
                f"checkpoint_step={g_step}, "
                f"next_global_step={global_step}, "
                f"resume_batch_idx={resume_batch_idx}"
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

        epoch_str = max(epoch_str, 1)
    except IndexError as e:
        # utils.latest_checkpoint_path() raises IndexError when
        # a fresh experiment has no matching checkpoint files.
        #
        # That specific case is allowed. If ANY local checkpoint
        # exists, an IndexError means the resume set is incomplete
        # or inconsistent and must fail instead of silently
        # restarting from step 0.
        _resume_files = []

        if os.path.isdir(
            hps.model_dir
        ):
            _resume_files = [
                name
                for name in os.listdir(
                    hps.model_dir
                )
                if (
                    name.endswith(".pth")
                    and name.startswith(
                        (
                            "G_",
                            "D_",
                            "DUR_",
                        )
                    )
                )
            ]

        if _resume_files:
            raise RuntimeError(
                "Resume discovery failed while "
                "checkpoint files are present: "
                + repr(
                    sorted(
                        _resume_files
                    )
                )
            ) from e

        if rank == 0:
            print(
                "No local resume checkpoints found; "
                "starting fresh."
            )

        epoch_str = 1
        global_step = 0
        resume_batch_idx = 0

    except Exception:
        # Corrupt checkpoints, incompatible state dictionaries,
        # optimizer errors, metadata inconsistencies, etc. must
        # never be converted into a fresh training run.
        raise

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        try:
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, eval_loader],
                    logger,
                    [writer, writer_eval],
                    resume_batch_idx if epoch == epoch_str else 0,
                )
            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                    resume_batch_idx if epoch == epoch_str else 0,
                )
        except Exception:
            # Training failures must propagate to the caller.
            # Swallowing an exception here makes failed runs
            # exit with status 0 and can incorrectly advance
            # LR schedulers despite no successful optimizer step.
            torch.cuda.empty_cache()
            raise

        _max_global_step_raw = os.environ.get(
            "MELOTTS_MAX_GLOBAL_STEP"
        )

        if _max_global_step_raw is not None:
            _max_global_step = int(
                _max_global_step_raw
            )

            if global_step >= _max_global_step:
                if rank == 0:
                    print(
                        "Training stopped cleanly at "
                        f"global_step={global_step}."
                    )
                return

        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    resume_batch_idx=0,
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    _max_global_step_raw = os.environ.get(
        "MELOTTS_MAX_GLOBAL_STEP"
    )

    if _max_global_step_raw is None:
        _max_global_step = None
    else:
        try:
            _max_global_step = int(
                _max_global_step_raw
            )
        except ValueError as e:
            raise RuntimeError(
                "MELOTTS_MAX_GLOBAL_STEP must be an integer, got: "
                + repr(_max_global_step_raw)
            ) from e

        if _max_global_step < 0:
            raise RuntimeError(
                "MELOTTS_MAX_GLOBAL_STEP must be >= 0"
            )

        if (
            _max_global_step
            % hps.train.eval_interval
            != 0
        ):
            raise RuntimeError(
                "MELOTTS_MAX_GLOBAL_STEP must be divisible by "
                "train.eval_interval so the stop step has a "
                "complete G/D/DUR checkpoint set: "
                f"max_step={_max_global_step}, "
                f"eval_interval={hps.train.eval_interval}"
            )

    net_g.train()

    _image_log_interval = int(
        getattr(
            hps.train,
            "image_log_interval",
            hps.train.log_interval,
        )
    )

    if _image_log_interval <= 0:
        raise RuntimeError(
            "train.image_log_interval must be > 0"
        )

    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        styles,
        tone,
        language,
        bert,
        ja_bert,
    ) in enumerate(tqdm(train_loader)):
        if batch_idx < resume_batch_idx:
            continue
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(
            rank,
            non_blocking=True,
        )
        styles = styles.cuda(
            rank,
            non_blocking=True,
        )
        tone = tone.cuda(rank, non_blocking=True)
        language = language.cuda(rank, non_blocking=True)
        bert = bert.cuda(rank, non_blocking=True)
        ja_bert = ja_bert.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
                style_id=styles,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                torch.nn.utils.clip_grad_norm_(
                net_dur_disc.parameters(),
                max_norm=10.0,
            ).item()
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = torch.nn.utils.clip_grad_norm_(
                net_d.parameters(),
                max_norm=10.0,
            ).item()
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = torch.nn.utils.clip_grad_norm_(
                net_g.parameters(),
                max_norm=10.0,
            ).item()
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [
                    loss_disc,
                    loss_gen,
                    loss_fm,
                    loss_mel,
                    loss_dur,
                    loss_kl,
                ]

                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch,
                        100.0
                        * batch_idx
                        / len(train_loader),
                    )
                )

                logger.info(
                    [x.item() for x in losses]
                    + [global_step, lr]
                )

                scalar_dict = {
                    "loss/g/total":
                        loss_gen_all,

                    "loss/d/total":
                        loss_disc_all,

                    "learning_rate":
                        lr,

                    "grad_norm_d":
                        grad_norm_d,

                    "grad_norm_g":
                        grad_norm_g,
                }

                scalar_dict.update(
                    {
                        "loss/g/fm":
                            loss_fm,

                        "loss/g/mel":
                            loss_mel,

                        "loss/g/dur":
                            loss_dur,

                        "loss/g/kl":
                            loss_kl,
                    }
                )

                scalar_dict.update(
                    {
                        "loss/g/{}".format(i): v
                        for i, v
                        in enumerate(losses_gen)
                    }
                )

                scalar_dict.update(
                    {
                        "loss/d_r/{}".format(i): v
                        for i, v
                        in enumerate(losses_disc_r)
                    }
                )

                scalar_dict.update(
                    {
                        "loss/d_g/{}".format(i): v
                        for i, v
                        in enumerate(losses_disc_g)
                    }
                )

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

            if (
                global_step
                % _image_log_interval
                == 0
            ):
                image_dict = {
                    "slice/mel_org":
                        utils.plot_spectrogram_to_numpy(
                            y_mel[0]
                            .data
                            .cpu()
                            .numpy()
                        ),

                    "slice/mel_gen":
                        utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0]
                            .data
                            .cpu()
                            .numpy()
                        ),

                    "all/mel":
                        utils.plot_spectrogram_to_numpy(
                            mel[0]
                            .data
                            .cpu()
                            .numpy()
                        ),

                    "all/attn":
                        utils.plot_alignment_to_numpy(
                            attn[0, 0]
                            .data
                            .cpu()
                            .numpy()
                        ),
                }

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                )

            _is_epoch_end = (
                batch_idx + 1 == len(train_loader)
            )

            _checkpoint_epoch_end = (
                os.environ.get(
                    "MELOTTS_CHECKPOINT_EPOCH_END",
                    "0",
                )
                == "1"
            )

            if (
                global_step % hps.train.eval_interval == 0
                or (
                    _checkpoint_epoch_end
                    and _is_epoch_end
                )
            ):
                if (
                    _checkpoint_epoch_end
                    and _is_epoch_end
                    and rank == 0
                ):
                    print(
                        "EPOCH_BOUNDARY_CHECKPOINT: "
                        f"epoch={epoch}, "
                        f"global_step={global_step}"
                    )

                if os.environ.get("MELOTTS_SKIP_EVAL", "0") != "1":
                    evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        if (
            _max_global_step is not None
            and global_step >= _max_global_step
        ):
            if rank == 0:
                print(
                    "MELOTTS_MAX_GLOBAL_STEP reached: "
                    f"{global_step}. "
                    "Checkpoint saved; ending current training run."
                )
            return

        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            styles,
            tone,
            language,
            bert,
            ja_bert,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            styles = styles.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    y=spec,
                    style_id=styles,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()
    print('Evauate done')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
