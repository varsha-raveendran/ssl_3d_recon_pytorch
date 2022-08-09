from src.evaluation import save_pcl 
import sys



def main():

    config = {
    'use_symmetry_loss': False,
     "use_pretrained" : True,
    'experiment_name' : 'chair_full_nosym',
    'device': 'cuda:0',  
    'is_overfit': True,
    'category' : '03001627',
    'batch_size': 2,
    'resume_ckpt': None,
    'learning_rate': 0.001,
    'max_epochs': 600,
    'print_every_n': 1,
    'validate_every_n': 1,
    'learning_rate_pose_net' : 0.0005,
    'learning_rate_recon_net' : 0.0005,
    'n_proj' : 5,
    'lambda_ae' : 100.0,
    'lambda_3d' : 500.0,
    'lambda_pose' : 1.0,
    'lambda_ae_mask' : 1000,
    'lambda_mask_fwd' : .00001,
    'lambda_mask_bwd' : .00001,
    'lambda_symm' : 0,
    'lambda_mask_pose' : 1.0,
    'lambda_ae_pose' : 1.0,
    "iso" : True
}

    save_pcl.main(config)

if __name__ == '__main__':
	sys.exit(main())
