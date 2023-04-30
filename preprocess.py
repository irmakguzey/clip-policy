import glob
import hydra
from omegaconf import DictConfig

from clip_policy.clip import Preprocessor
import sys


@hydra.main(
    version_base=None, config_path="clip_policy/configs", config_name="preprocess"
)
def main(cfg: DictConfig) -> None:
    # sys.path.insert(0, "/scratch/ar7420/VINN/Detic/third_party/CenterNet2/")

    prep = Preprocessor(
        env_path=cfg.env_path,
        score_thresh_test=cfg.classifier_threshold,
        classes=cfg.classifier_classes,
        visualize=cfg.visualize
    )

    prep.process_env()


if __name__ == "__main__":
    main()
