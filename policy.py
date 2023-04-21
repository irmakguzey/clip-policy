import hydra

from omegaconf import DictConfig

from clip_policy.clip import *
from clip_policy.models import *
from clip_policy.utils import *

@hydra.main(version_base=None, config_path='clip_policy/configs', config_name='policy')
def main(cfg : DictConfig) -> None:
    
    # Get the query vocab

    # Get the embeddings from the received query

    # Sort the datasets

    # Get the VINN action in a loop with the given dataloaders


if __name__ == '__main__':
    main()