import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.diffpose_frame import Diffpose


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=19960903, help="Random seed")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the config file")
    parser.add_argument("--exp", type=str, default="exp", 
                        help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, required=True, 
                        help="A string for documentation purpose. "\
                            "Will be the name of the log folder.", )
    parser.add_argument("--verbose", type=str, default="info", 
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    ### Diffformer configuration ####
    #Diffusion process hyperparameters
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quad(quadratic))")
    parser.add_argument("--eta", type=float, default=0.0, 
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")
    # Diffusion model parameters
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor')
    # load pretrained model
    parser.add_argument('--model_diff_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--model_pose_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--train', action = 'store_true',
                        help='train or evluate')
    #training hyperparameter
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--lr_gamma', default=0.9, type=float, metavar='N',
                        help='weight decay rate')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--decay', default=60, type=int, metavar='N',
                        help='decay frequency(epoch)')
    #test hyperparameter
    parser.add_argument('--test_times', default=5, type=int, metavar='N',
                    help='the number of test times')
    parser.add_argument('--test_timesteps', default=50, type=int, metavar='N',
                    help='the number of test time steps')
    parser.add_argument('--test_num_diffusion_timesteps', default=500, type=int, metavar='N',
                    help='the number of test times')
                        
    # Add implicit layer arguments
    parser.add_argument('--implicit_layers', action='store_true',
                        help='Use implicit layers in the model')
    parser.add_argument('--implicit_start', type=int, default=4,
                        help='Start implicit layers from this index (0-indexed)')
    parser.add_argument('--implicit_max_iter', type=int, default=1,
                        help='Maximum iterations for implicit layers')
    parser.add_argument('--implicit_max_iter_final', type=int, default=5,
                        help='Final maximum iterations to reach')
    parser.add_argument('--implicit_tol', type=float, default=0.05,
                        help='Convergence tolerance for implicit layers')
    parser.add_argument('--implicit_warmup_epochs', type=int, default=60,
                        help='Number of epochs to run with minimal iterations')
    parser.add_argument('--enable_warmstart', action='store_true',
                        help='Enable warm starting between diffusion steps')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extra safety checks')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device
    
    # update configure file
    new_config.training.batch_size = args.batch_size
    new_config.optim.lr = args.lr
    new_config.optim.lr_gamma = args.lr_gamma
    new_config.optim.decay = args.decay
    
    # Add implicit configuration to config if specified in args
    if args.implicit_layers:
        new_config.model.implicit_layers = True
        new_config.model.implicit_start_layer = args.implicit_start
        new_config.model.implicit_max_iter = args.implicit_max_iter
        new_config.model.implicit_max_iter_final = args.implicit_max_iter_final
        new_config.model.implicit_tol = args.implicit_tol
        new_config.training.implicit_warmup_epochs = args.implicit_warmup_epochs
    
    # Set mixed precision and warmstart if specified
    if args.mixed_precision:
        new_config.training.mixed_precision = True
    
    if args.enable_warmstart:
        new_config.testing.enable_warmstart = True
        
    # Add testing configuration
    new_config.testing.test_times = args.test_times
    new_config.testing.test_timesteps = args.test_timesteps
    new_config.testing.test_num_diffusion_timesteps = args.test_num_diffusion_timesteps

    if args.train:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        # Clear existing handlers to prevent duplication
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    
    # Set deterministic behavior for reproducibility if debugging
    if args.debug:
        logging.info("Setting deterministic behavior for debugging")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    # Track GPU memory status
    if torch.cuda.is_available():
        logging.info("GPU Memory: {:.2f} GB free out of {:.2f} GB total".format(
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3),
            torch.cuda.get_device_properties(0).total_memory / (1024**3)))
    
    try:
        runner = Diffpose(args, config)
        runner.create_diffusion_model(args.model_diff_path)
        runner.create_pose_model(args.model_pose_path)
        runner.prepare_data()
        
        # Force explicit garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU Memory after model creation: {:.2f} GB free out of {:.2f} GB total".format(
                (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3),
                torch.cuda.get_device_properties(0).total_memory / (1024**3)))
        
        # Train or evaluate based on arguments
        if args.train:
            runner.train()
        else:
            _, _ = runner.test_hyber()
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())