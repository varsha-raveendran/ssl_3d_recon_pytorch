from src.training import train_full
import sys



def main():

    config = {
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': True,
    'category' : '02691156',
    'batch_size': 1,
    'resume_ckpt': None,
    'learning_rate': 0.0005,
    'max_epochs': 60,
    'print_every_n': 1,
    'validate_every_n': 1,
    'learning_rate_pose_net' : 0.0001,
    'learning_rate_recon_net' : 0.0001,
    'n_proj' : 3,
    'lambda_ae' : 1,
    'lambda_3d' : 1,
    'lambda_pose' : 1,
    'lambda_ae_mask' : 1,
    'lambda_mask_fwd' : 1,
    'lambda_mask_bwd' : 1,
    'lambda_symm' : 1,
    'lambda_mask_pose' : 1
    }


    train_full.main(config)

if __name__ == '__main__':
	sys.exit(main())