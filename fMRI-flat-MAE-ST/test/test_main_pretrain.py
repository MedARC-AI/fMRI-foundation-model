import shutil
from pathlib import Path

from main_pretrain import main, get_args_parser


def test_main():
    output_dir = Path("test_output/test_default")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    parser = get_args_parser()
    args = parser.parse_args(
        [
            "--model",
            "mae_vit_small_patch16_fmri",
            "--output_dir",
            "test_output",
            "--name",
            "test_default",
            # "--wandb",
            "--debug",
        ]
    )
    main(args)
