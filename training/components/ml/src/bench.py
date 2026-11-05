import hydra
import subprocess
import os
from omegaconf import DictConfig
from tqdm import tqdm

def do_train(cfg: DictConfig):

    validation = cfg.validation if 'validation' in cfg else False
    print("Training classifiers ...")
    print(f'Validation: {validation}')
    with tqdm(total = sum(map(lambda item: len(item.classifiers),cfg.train))) as pbar:
        for experiment in cfg.train:
            dataset = experiment.dataset
            sweep = experiment.sweep
            train_cfg = experiment.train_cfg
            for classifier in experiment.classifiers:        
                pbar.set_description(f"Training classifier: {classifier} on {dataset}...")
                path = os.path.join(hydra.utils.get_original_cwd(),'train.py')
                cmd = f'{cfg.environment.python} {path} -m --config-name {train_cfg} +sweep/{sweep}@_global_={classifier} dataset={dataset}'
                if validation: cmd += " sweep.budget=1"
                subprocess.run(
                    cmd
                )
                pbar.update()

def do_test(cfg: DictConfig):
    
    print("Testing classifiers ...")    
    with tqdm(total = sum(map(lambda item: len(item.classifiers),cfg.test))) as pbar:
        for experiment in cfg.test:
            dataset = experiment.dataset           
            test_cfg = experiment.test_cfg
            for classifier in experiment.classifiers:        
                pbar.set_description(f"Testing classifier: {classifier} on {dataset}...")
                path = os.path.join(hydra.utils.get_original_cwd(),'test.py')
                cmd = f'{cfg.environment.python} {path} --config-name {test_cfg} +classifier={classifier} +clfparameters/{dataset}@_global_={classifier} +dataset={dataset}'
                subprocess.run(
                    cmd
                )
                pbar.update()

@hydra.main(version_base = None, config_path='.conf',config_name='bench')
def main(cfg : DictConfig):
    if 'train' in cfg:
        do_train(cfg)
    if 'test' in cfg:
        do_test(cfg)
    
if __name__ == "__main__":
    main()