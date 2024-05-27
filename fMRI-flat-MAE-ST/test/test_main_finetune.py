import shutil
from pathlib import Path

from main_finetune import main, get_args_parser


def test_main():
    output_dir = Path("test_output/test_main_finetune")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    ckpt_path = Path(__file__).parents[1] / "output/pretrain_01/checkpoint-00099.pth"

    parser = get_args_parser()
    args = parser.parse_args(
        [
            "--batch_size",
            "8",
            "--epochs",
            "1",
            "--model",
            "vit_small_patch16_fmri",
            "--warmup_epochs",
            "1",
            "--finetune",
            str(ckpt_path),
            "--freeze_params",
            "*",
            "--unfreeze_params",
            "blocks.*.attn.*,norm.*,spatial_pool.*,head.*",
            "--global_pool",
            "spatial",
            "--num_train_samples",
            "256",
            "--num_val_samples",
            "256",
            "--output_dir",
            "test_output",
            "--name",
            "test_main_finetune",
            # "--wandb",
        ]
    )
    main(args)
