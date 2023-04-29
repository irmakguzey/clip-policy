import hydra
from robot.controller import Controller
import torch
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="run")
def main(cfg):

    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    controller = Controller(cfg = dict_cfg)
    
    if cfg['model_type'] == 'vinn':
        model = hydra.utils.instantiate(cfg.model)()
        model.k = cfg["k"] # NOTE: There is a way better way to do that
        model.bs = cfg["bs"]
        model.load_state_variables(cfg["model_pth"])
    else:
        model = hydra.utils.instantiate(cfg.model)(cfg=cfg)
        model.load_state_dict(torch.load(cfg["model_pth"]))
        model.eval()
    
    controller.setup_model(model)
    controller.run()

if __name__ == "__main__":

    main()
