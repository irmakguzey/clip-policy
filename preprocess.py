import glob
import hydra 
from omegaconf import DictConfig

from clip_policy.clip_preprocess import Preprocessor

@hydra.main(version_base=None, config_path='clip_policy/configs', config_name='preprocess')
def main(cfg : DictConfig) -> None:
    
    prep = Preprocessor(
        env_path = cfg.env_path,
        score_thresh_test = cfg.classifier_threshold,
        classes = cfg.classifier_classes
    )

    prep.process_env()

if __name__ == '__main__':
    main()