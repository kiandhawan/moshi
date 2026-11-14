import logging
import time
from typing import Iterator

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
):
    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)

    text_loss = torch.tensor(0.0).cuda()
    audio_loss = torch.tensor(0.0).cuda()
    model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Take the batches first so every rank runs the same number of forward passes: with
    # FSDP each forward is a collective, and a rank with one batch fewer would hang the rest.
    batches = []
    for batch in eval_data_loader:
        if len(batches) >= max(1, 40 // get_world_size()):
            break
        batches.append(batch)
    n_batches = torch.tensor([len(batches)], device="cuda", dtype=torch.long)
    dist.all_reduce(n_batches, op=dist.ReduceOp.MIN)
    if n_batches.item() < len(batches):
        main_logger_info(f"Eval: using {n_batches.item()} batches per rank (ranks differ).")
    for batch in batches[: n_batches.item()]:
        num_samples += 1
        with torch.no_grad():
            codes = batch.codes
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            output = model(codes=codes, condition_tensors=condition_tensors)
            text_loss += compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss += compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )
    eval_loss = text_loss + audio_loss
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_samples = int(torch.tensor(all_num_samples).sum().item())
    # sum loss
    torch.cuda.synchronize()
    main_logger_info(f"Eval finished! {int(total_num_samples)} batches over {get_world_size()} ranks "
                     f"in {time.perf_counter() - t0:.1f} s (data loading + forward passes)")

    dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)
    text_loss /= total_num_samples
    audio_loss /= total_num_samples
    eval_loss /= total_num_samples

    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()
    state.this_audio_loss = audio_loss.item()
    state.this_text_loss = text_loss.item()

    # train mode!
    model.train()
