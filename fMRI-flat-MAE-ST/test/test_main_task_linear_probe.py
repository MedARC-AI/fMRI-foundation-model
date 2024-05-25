from pathlib import Path

from main_task_linear_probe import main, get_args_parser


def test_main():
    output_path = Path("test_output/test_task_linear_probe/result.json")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feat_prefix = Path(__file__).parents[1] / "output/vit_small_patch16_fmri_random_features"
    parser = get_args_parser()
    args = parser.parse_args(
        [
            "--output_path",
            str(output_path),
            "--feat_prefix",
            str(feat_prefix),
        ]
    )
    main(args)
