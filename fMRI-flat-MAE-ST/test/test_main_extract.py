import pytest
from pathlib import Path

from main_extract import main, get_args_parser


@pytest.mark.parametrize("clip_mode", ["seq", "event"])
def test_main(clip_mode: str):
    output_path = Path("test_output/test_extract/pretrain_01.parquet")
    if output_path.exists():
        output_path.unlink()

    ckpt_path = Path(__file__).parents[1] / "output/pretrain_01/checkpoint-00099.pth"
    parser = get_args_parser()
    args = parser.parse_args(
        [
            "--output_path",
            str(output_path),
            "--model",
            "vit_small_patch16_fmri",
            "--ckpt_path",
            str(ckpt_path),
            "--clip_mode",
            clip_mode,
            "--num_samples",
            "32",
            "--device",
            "cpu",
        ]
    )
    main(args)
