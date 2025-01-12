import os
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import wandb
from transformers import AutoTokenizer, CLIPTextModel
import webdataset as wds
from webdataset.handlers import warn_and_continue
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtools.utils import Diffuzz
from vqgan import VQModel
from modules import DiffNeXt, EfficientNetEncoder, Prior
from utils import transforms, effnet_preprocess, identity
import transformers
from transformers.utils import is_torch_bf16_available, is_torch_tf32_available


transformers.utils.logging.set_verbosity_error()

# PARAMETERS
updates = 10000
warmup_updates = 1000
ema_start = 0
ema_every = 100
ema_beta = 0.9
batch_size = 8
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 250 * grad_accum_steps
extra_ckpt_every = 1000 * grad_accum_steps
lr = 1e-4

dataset_path = "pokemon.tar"
run_name = "poke"
output_path = f"output/{run_name}"
os.makedirs(output_path, exist_ok=True)
checkpoint_dir = f"checkpoints/"
checkpoint_path = os.path.join(checkpoint_dir, run_name, "model.pt")
os.makedirs(os.path.join(checkpoint_dir, run_name), exist_ok=True)
# model to fine tune from
load_from = "models/model_v2_stage_c_finetune_interpolation.pt"

# pretrained models
vq_model_path = "models/vqgan_f4_v1_500k.pt"
stage_b_model_path = "models/model_v2_stage_b.pt"

wandb_project = "w2"
wandb_entity = "justinpinkney"
wandb_run_name = run_name

val_caps = [
    "Hello Kitty",
    "An old man looking at the moon",
    "Donald Trump",
    "Yoda",
    "A hungry dog",
    "A watermelon",
    "Bowl of delicous soup"
    "Pikachu",
]


def train(n_nodes=1):
    # assuming 1 node here
    gpu_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(gpu_id)
    main_node = gpu_id == 0
    device = torch.device(gpu_id)
    init_process_group(backend="nccl")

    # only ampere gpu architecture allows these
    _float16_dtype = torch.float16 if not is_torch_bf16_available() else torch.bfloat16
    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- PREPARE DATASET ---
    dataset = (
        wds.WebDataset(dataset_path, resampled=True, handler=warn_and_continue)
        .shuffle(44, handler=warn_and_continue)
        .decode("pilrgb", handler=warn_and_continue)
        .to_tuple("jpg", "txt", handler=warn_and_continue)
        .map_tuple(transforms, identity, handler=warn_and_continue)
    )

    real_batch_size = batch_size // (world_size * n_nodes * grad_accum_steps)
    dataloader = DataLoader(
        dataset, batch_size=real_batch_size, num_workers=1, pin_memory=False
    )

    if main_node:
        print("REAL BATCH SIZE / DEVICE:", real_batch_size)

    checkpoint = torch.load(load_from, map_location=device)

    # - vqmodel -
    if main_node:
        vqmodel = VQModel().to(device)
        vqmodel.load_state_dict(
            torch.load(vq_model_path, map_location=device)["state_dict"]
        )
        vqmodel.eval().requires_grad_(False)

    diffuzz = Diffuzz(device=device)

    # - CLIP text encoder
    clip_model = (
        CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )

    clip_model_b = (
        CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    clip_tokenizer_b = AutoTokenizer.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )

    # - EfficientNet -
    pretrained_checkpoint = torch.load(stage_b_model_path, map_location=device)

    effnet = EfficientNetEncoder().to(device)
    effnet.load_state_dict(pretrained_checkpoint["effnet_state_dict"])
    effnet.eval().requires_grad_(False)
    # - Paella Model as generator -
    if main_node:
        generator = DiffNeXt().to(device)
        generator.load_state_dict(pretrained_checkpoint["state_dict"])
        generator.eval().requires_grad_(False)

    del pretrained_checkpoint

    # v2 prior
    model = Prior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(device)
    if checkpoint is not None:
        # v2 checkpoint only has ema state
        model.load_state_dict(checkpoint["ema_state_dict"])

    if main_node:  # <--- DDP
        model_ema = (
            Prior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24)
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    # load checkpoints & prepare ddp
    if checkpoint is not None:
        if main_node:  # <--- DDP
            model_ema.load_state_dict(checkpoint["ema_state_dict"])

    # - SETUP WANDB -
    if main_node:  # <--- DDP
        run_id = wandb.util.generate_id()
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            id=run_id,
            resume="allow",
        )

    model = DDP(model, device_ids=[gpu_id], output_device=device)  # <--- DDP

    if main_node:  # <--- DDP
        print(
            "Num trainable params:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # eps=1e-4
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_updates
    )
    scaler = torch.cuda.amp.GradScaler()
    start_iter = 1
    grad_norm = torch.tensor(0, device=device)
    ema_loss = None

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

    if main_node:
        print("Everything prepared, starting training now....")
    dataloader_iterator = iter(dataloader)
    pbar = (
        tqdm(range(start_iter, max_iters + 1))
        if (main_node)
        else range(start_iter, max_iters + 1)
    )  # <--- DDP
    model.train()
    for it in pbar:
        bls = time.time()
        images, captions = next(dataloader_iterator)
        ble = time.time() - bls
        images = images.to(device)

        with torch.no_grad():
            effnet_features = effnet(effnet_preprocess(images))
            with torch.cuda.amp.autocast(dtype=_float16_dtype):
                if (
                    np.random.rand() < 0.05
                ):  # 90% of the time, drop the CLIP text embeddings (independently)
                    clip_captions = [""] * len(
                        captions
                    )  # 5% of the time drop all the captions
                else:
                    clip_captions = captions
                clip_tokens = clip_tokenizer(
                    clip_captions,
                    truncation=True,
                    padding="max_length",
                    max_length=clip_tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

            t = (
                (1 - torch.rand(images.size(0), device=device))
                .mul(1.08)
                .add(0.001)
                .clamp(0.001, 1.0)
            )
            noised_embeddings, noise = diffuzz.diffuse(effnet_features, t)

        with torch.cuda.amp.autocast(dtype=_float16_dtype):
            pred_noise = model(noised_embeddings, t, clip_text_embeddings)
            loss = nn.functional.mse_loss(pred_noise, noise, reduction="none").mean(
                dim=[1, 2, 3]
            )
            loss_adjusted = (loss * diffuzz.p2_weight(t)).mean() / grad_accum_steps

        if it % grad_accum_steps == 0 or it == max_iters:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if main_node and (it % ema_every == 0 or it == max_iters):
                if it < ema_start:
                    model_ema.load_state_dict(model.module.state_dict())
                else:
                    model_ema.update_weights_ema(model.module, beta=ema_beta)
        else:
            with model.no_sync():
                loss_adjusted.backward()

        ema_loss = (
            loss.mean().item()
            if ema_loss is None
            else ema_loss * 0.99 + loss.mean().item() * 0.01
        )

        if main_node:
            pbar.set_postfix(
                {
                    "bs": images.size(0),
                    "batch_loading": ble,
                    "loss": loss.mean().item(),
                    "loss_adjusted": loss_adjusted.item(),
                    "ema_loss": ema_loss,
                    "grad_norm": grad_norm.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "total_steps": scheduler.last_epoch,
                }
            )

        if main_node:
            wandb.log(
                {
                    "loss": loss.mean().item(),
                    "loss_adjusted": loss_adjusted.item(),
                    "ema_loss": ema_loss,
                    "grad_norm": grad_norm.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "total_steps": scheduler.last_epoch,
                }
            )

        if main_node and (
            it == 1 or it % print_every == 0 or it == max_iters
        ):  # <--- DDP
            tqdm.write(f"ITER {it}/{max_iters} - loss {ema_loss}")

            if it % extra_ckpt_every == 0:
                torch.save(
                    {
                        "state_dict": model.module.state_dict(),
                        "ema_state_dict": model_ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_last_step": scheduler.last_epoch,
                        "iter": it,
                        "metrics": {
                            "ema_loss": ema_loss,
                        },
                        "grad_scaler_state_dict": scaler.state_dict(),
                        "wandb_run_id": run_id,
                    },
                    os.path.join(checkpoint_dir, run_name, f"model_{it}.pt"),
                )

            torch.save(
                {
                    "state_dict": model.module.state_dict(),
                    "ema_state_dict": model_ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_last_step": scheduler.last_epoch,
                    "iter": it,
                    "metrics": {
                        "ema_loss": ema_loss,
                    },
                    "grad_scaler_state_dict": scaler.state_dict(),
                    "wandb_run_id": run_id,
                },
                checkpoint_path,
            )

            model.eval()
            images, captions = next(dataloader_iterator)
            images, captions = images.to(device), captions

            limit_n_val_images = 8  # in case there is a big batch size
            images = images[:limit_n_val_images]
            captions = val_caps[:len(images)]
            with torch.no_grad():
                clip_tokens = clip_tokenizer(
                    captions,
                    truncation=True,
                    padding="max_length",
                    max_length=clip_tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

                clip_tokens_uncond = clip_tokenizer(
                    [""] * len(captions),
                    truncation=True,
                    padding="max_length",
                    max_length=clip_tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                clip_text_embeddings_uncond = clip_model(
                    **clip_tokens_uncond
                ).last_hidden_state

                t = (
                    (1 - torch.rand(images.size(0), device=device))
                    .add(0.001)
                    .clamp(0.001, 1.0)
                )
                effnet_features = effnet(effnet_preprocess(images))
                effnet_features = effnet_features.mul(42).sub(1)
                effnet_embeddings_uncond = torch.zeros_like(effnet_features)
                # noised_embeddings, noise = diffuzz.diffuse(effnet_features, t)

                with torch.cuda.amp.autocast(dtype=_float16_dtype):
                    sampled = diffuzz.sample(
                        model.module,
                        {"c": clip_text_embeddings},
                        unconditional_inputs={"c": clip_text_embeddings_uncond},
                        shape=effnet_features.shape,
                        cfg=6,
                    )[-1]
                    sampled_ema = diffuzz.sample(
                        model_ema,
                        {"c": clip_text_embeddings},
                        unconditional_inputs={"c": clip_text_embeddings_uncond},
                        shape=effnet_features.shape,
                        cfg=6,
                    )[-1]

                    # scaling required for v2
                    sampled = sampled.mul(42).sub(1)
                    sampled_ema = sampled_ema.mul(42).sub(1)

                    # second clip
                    clip_text_embeddings = clip_model_b(**clip_tokens).last_hidden_state
                    clip_tokens_uncond = clip_tokenizer_b(
                        [""] * len(captions),
                        truncation=True,
                        padding="max_length",
                        max_length=clip_tokenizer_b.model_max_length,
                        return_tensors="pt",
                    ).to(device)
                    clip_text_embeddings_uncond = clip_model_b(
                        **clip_tokens_uncond
                    ).last_hidden_state
                    sampled_images = diffuzz.sample(
                        generator,
                        {"effnet": sampled_ema, "clip": clip_text_embeddings},
                        (
                            clip_text_embeddings.size(0),
                            4,
                            images.size(-2) // 4,
                            images.size(-1) // 4,
                        ),
                        unconditional_inputs={
                            "effnet": effnet_embeddings_uncond,
                            "clip": clip_text_embeddings_uncond,
                        },
                    )[-1]
                    sampled_images_ema = diffuzz.sample(
                        generator,
                        {"effnet": sampled, "clip": clip_text_embeddings},
                        (
                            clip_text_embeddings.size(0),
                            4,
                            images.size(-2) // 4,
                            images.size(-1) // 4,
                        ),
                        unconditional_inputs={
                            "effnet": effnet_embeddings_uncond,
                            "clip": clip_text_embeddings_uncond,
                        },
                    )[-1]
                sampled_images = vqmodel.decode(sampled_images).clamp(0, 1)
                sampled_images_ema = vqmodel.decode(sampled_images_ema).clamp(0, 1)

                torchvision.utils.save_image(
                    torch.cat(
                        [
                            torch.cat([i for i in sampled_images.cpu()], dim=-1),
                            torch.cat([i for i in sampled_images_ema.cpu()], dim=-1),
                        ],
                        dim=-2,
                    ),
                    f"{output_path}/{it:06d}.jpg",
                )
            model.train()

            log_data = [
                [captions[i]]
                + [wandb.Image(sampled_images[i])]
                + [wandb.Image(sampled_images_ema[i])]
                for i in range(len(images))
            ]
            log_table = wandb.Table(
                data=log_data,
                columns=["Captions", "Sampled", "Sampled EMA"],
            )
            wandb.log({"Log": log_table})
            del (
                clip_tokens,
                clip_text_embeddings,
                clip_tokens_uncond,
                clip_text_embeddings_uncond,
                t,
                effnet_features,
                effnet_embeddings_uncond,
                noised_embeddings,
                noise,
                pred_noise,
                sampled,
                sampled_ema,
                sampled_images,
                sampled_images_ema,
            )
            del log_data, log_table

    destroy_process_group()  # <--- DDP


if __name__ == "__main__":
    train()
    # to train on 2 devices: torchrun --nnodes=1 --nproc-per-node=2 train_stage_C.py
