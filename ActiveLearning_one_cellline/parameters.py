args_pool = {'LINCS':
				{'n_epoch': 20, 
				 'name': 'LINCS',
                 'cell': ['MCF7', 'A549', 'PC3'],
                #  'balancesample': False,
                #  'balancesample': True,
                #  'transform_train': 'no',
				#  'transform': 'no',
				 'loader_tr_args':{'batch_size': 32, 'num_workers': 4},
				 'loader_te_args':{'batch_size': 256, 'num_workers': 4},
				 'num_class':3,
				 'optimizer':'Adam',
				 'pretrained': True,
				 'optimizer_args':{'lr': 0.001}},
}