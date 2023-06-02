import argparse
class config():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='train', help='train | test | valid')
        parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
        parser.add_argument('--gpu_id', type=str, default='0', choices=('0', '1', '2', '3'))
        parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        parser.add_argument('--niter', type=int, default=50, help='of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=150, help='of iter to linearly decay learning rate to zero')
        parser.add_argument('--epoch_count', type=int, default=1, help='number of epochs to train begin for')
        parser.add_argument('--modelLR', type=float, default=1e-4, help='learning rate for model')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--use_dropout', type=bool, default=True,
                            help='use dropout | True | False')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--modelWeights', type=str, default='',
                            help="path to model weights (to continue training)")
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--out', type=str, default='./checkpoint',
                            help='folder to output model checkpoints')
        parser.add_argument('--root', type=str, default='./CentralCrop',help='folder to dataloader')
        parser.add_argument('--root0', type=str, default='./finalclassification-raw/finalclassification-raw/FDG-nii-raw-MELANOMA',help='folder to dataloader')
        parser.add_argument('--root1', type=str, default='./finalclassification-raw/finalclassification-raw/FDG-nii-raw-LYMPHOMA',help='folder to dataloader')
        parser.add_argument('--root2', type=str, default='./finalclassification-raw/finalclassification-raw/FDG-nii-raw-LUNG_CANCER',help='folder to dataloader')
        parser.add_argument('--root3', type=str, default='./finalclassification-raw/finalclassification-raw/FDG-nii-raw-age0_55',help='folder to dataloader')
        # parser.add_argument('--root', type=str, default='./test_data1',
        #                     help='folder to dataloader')
        parser.add_argument('--ratio', type=int, default=32)
        parser.add_argument('--weight', type=float,default=0.5)

        # parser.add_argument('--root4', type=str, default='./FDG-nii-raw-MELANOMA',
                            # help='folder to dataloader')
        # # parser.add_argument('--root', type=str, default='./test_data1',
        #                     help='folder to dataloader')
        parser.add_argument('--dataset_named', type=str, default='petct',choices=('petct', 'ct', 'pet'))
        # parser.add_argument('--setting', type=str,default='FL',choices=('central', 'FL'))
        
        # our method
        parser.add_argument('--setting', type=str,default='FL',choices=('central','comed', 'FL','FL_re_cosine','FL_re_distance','QFFedAvg','CGSV','ours'))
        parser.add_argument('--low_bound_ensure', type=int,default=0)
        parser.add_argument('--ours_dir', type=int,default=0)
        parser.add_argument('--re_cosine', type=int,default=0)
        parser.add_argument('--pccs', type=int,default=0)
        parser.add_argument('--re_pccs', type=int,default=0)
        parser.add_argument('--q', type=float,default=0.5)
        parser.add_argument('--ours_dis', type=int,default=0)
        parser.add_argument('--ours_loss', type=int,default=0)
        parser.add_argument('--ours_pccs', type=int,default=0)
        parser.add_argument('--min_threshold', type=float,default=0.01)
        
        parser.add_argument('--model_choice', type=str,default='best_model_embeding_attention',choices=('navie_model_concat', 'best_model_embeding_attention'))
        parser.add_argument('--backend', type=str,default="ncll")
        parser.add_argument('--init_method', type=str,default=None)
        parser.add_argument('--world_size', type=int,default=1)
        parser.add_argument('--rank', type=int,default=0)
        parser.add_argument('--DP', type=str,default='yes')
        parser.add_argument('--dp_delta', type=float,default=0.0)
        parser.add_argument('--standalone', type=str,default='false',choices=('false', 'true'))
        parser.add_argument('--noniid', type=str,default='noniid',choices=('noniid', 'iid'))
        parser.add_argument('--dataset_choice', type=str,default='age0_55', choices=('all','age0_55', 'age55_65', 'age65_above', 'LUNG_CANCER', 'LYMPHOMA', 'MELANOMA', 'NEGATIVE','female', 'male','na'))

        self.opt = parser.parse_args()

    def forward(self):
        return self.opt
    def __call__(self, *input, **kwargs):
         result = self.forward(*input, **kwargs)
         return result

opt= config()
print(opt())
