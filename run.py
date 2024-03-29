from src.training import train_v2
import sys



def main():

    config = {
    'iso': False,
    'use_symmetry_loss': False,
     "use_pretrained" : False,
    'experiment_name' : 'chair_full_nosym',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': False,
    'category' : '03001627',
    'batch_size': 2,
    'resume_ckpt': None,
    'learning_rate': 0.001,
    'max_epochs': 160,
    'print_every_n': 1,
    'validate_every_n': 1,
    'learning_rate_pose_net' : 0.0005,
    'learning_rate_recon_net' : 0.0005,
    'n_proj' : 5,
    'lambda_ae' : 100.0,
    'lambda_3d' : 10000.0,
    'lambda_pose' : 1.0,
    'lambda_ae_mask' : 1000,
    'lambda_mask_fwd' : 1e-5,
    'lambda_mask_bwd' : 1e-5,
    'lambda_symm' : 10,
    'lambda_mask_pose' : 1.0,
    'lambda_ae_pose' : 1.0
    }

    train_v2.main(config)

if __name__ == '__main__':
	sys.exit(main())
