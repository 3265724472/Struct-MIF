import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from torch.cuda.amp import GradScaler, autocast

# 保持你原来的引用！不要改这里
from src.modeling.struct_mif import StructMIF
from src.data.dataset import StructMIFDataset
from src.data.collator import StructMIFCollator
from src.loss import MaskedMLMLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Struct-MIF Pretraining")
    parser.add_argument("--train_path", type=str, required=True, help="Path to processed train graphs (.pt)")
    parser.add_argument("--esm_model_path", type=str, required=True, help="Path to ESM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)

    # 🔴 [修改] 增加新参数
    parser.add_argument("--gvp_layers", type=int, default=6, help="Depth")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Width (GNN hidden dim)")
    parser.add_argument("--top_k", type=int, default=30, help="Topology (0=use default, >0=dynamic knn)")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_3di"])
    parser.add_argument("--gnn_type", type=str, default="gvp", choices=["gvp", "gcn", "gat", "egnn"])
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Training on device: {device}")

    logger.info("Loading dataset...")
    # 使用原来的 StructMIFDataset
    dataset = StructMIFDataset(root_dir=args.train_path, tokenizer_name=args.esm_model_path)
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info(f"Initializing model (GNN={args.gnn_type}, K={args.top_k}, Dim={args.hidden_dim})...")

    # 🔴 [修改] 传入新参数
    model = StructMIF(
        esm_model_name=args.esm_model_path,
        gvp_layers=args.gvp_layers,
        gvp_node_out_dim=args.hidden_dim,  # 传入 hidden_dim
        top_k=args.top_k,  # 传入 top_k
        ablation_mode=args.ablation,
        gnn_type=args.gnn_type
    )

    model.to(device)
    model.train()

    # 使用原来的 StructMIFCollator (假设它在你本地 src/data/collator.py 里)
    collator = StructMIFCollator(tokenizer=dataset.tokenizer, max_length=1024)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = MaskedMLMLoss(ignore_index=-100)
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    logger.info("Start training...")

    for epoch in range(start_epoch, args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)

            with autocast():
                logits = model(batch)
                loss = criterion(logits, batch.y)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accum_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * args.accum_steps:.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)

        best_path = os.path.join(args.output_dir, "best_checkpoint.pt")
        torch.save(model.state_dict(), best_path)


if __name__ == "__main__":
    main()