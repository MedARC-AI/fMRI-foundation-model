import pytest
from pathlib import Path

from main_task_linear_probe import main, get_args_parser


@pytest.mark.parametrize("target", ["task", "trial_type"])
def test_main(target: str):
    output_dir = Path("test_output/test_task_linear_probe")

    feat_prefix = Path(__file__).parents[1] / "output/vit_small_patch16_fmri_random_features"
    if target == "trial_type":
        feat_prefix = f"{feat_prefix}_event"

    parser = get_args_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            str(output_dir),
            "--feat_prefix",
            str(feat_prefix),
            "--target",
            target,
        ]
    )
    main(args)
