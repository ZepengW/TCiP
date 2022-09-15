import logging
import torch
import torch.nn as nn
import copy
from .backbones.vit_pytorch_transreid import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID, trunc_normal_
from loss.triplet_loss import WeightedRegularizedTriplet, TripletLoss, CrossEntropyLabelSmooth
import model.net.backbones.vit_pytorch_transreid_mask as vit_mask


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TransReID(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TransReID, self).__init__()
        model_path = kwargs['PRETRAIN_PATH']
        pretrain_choice = kwargs['PRETRAIN_CHOICE']
        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'before'
        self.in_planes = 768
        transformer_type = kwargs['TRANSFORMER_TYPE']

        print('using Transformer_type: {} as a backbone'.format(transformer_type))
        camera_num = kwargs['CAMERA_NUM']
        view_num = 0
        factory ={
            'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
            'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
            'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
            'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
        }
        self.base = factory[transformer_type](img_size=kwargs['SIZE_INPUT'],
                                              sie_xishu=kwargs['SIE_COE'],
                                              local_feature=kwargs['JPM'],
                                              camera=camera_num,
                                              view=view_num,
                                              stride_size=kwargs['STRIDE_SIZE'],
                                              drop_path_rate=0.1,
                                              drop_rate=0.0,
                                              attn_drop_rate=0.0)
        if transformer_type == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = 2
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = 5
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = 4
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = kwargs['RE_ARRANGE']

    def forward(self, img, camera_id= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(img, cam_label=camera_id, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            img = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            img = features[:, 1:]
        # lf_1
        b1_local_feat = img[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = img[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = img[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = img[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_1 = self.classifier_1(local_feat_1_bn)
            cls_score_2 = self.classifier_2(local_feat_2_bn)
            cls_score_3 = self.classifier_3(local_feat_3_bn)
            cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class TCiP(nn.Module):
    def __init__(self, num_classes, num_clothes, feat_weight = [1.0, 1.0], use_mask = False, normal_scale = -1,
        xent_type = 'xent', use_pid_as_cloid = False, thred = 0, **kwargs):
        super(TCiP, self).__init__()
        model_path = kwargs['PRETRAIN_PATH']
        pretrain_choice = kwargs['PRETRAIN_CHOICE']
        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'before'
        self.in_planes = 768
        transformer_type = kwargs['TRANSFORMER_TYPE']

        print('using Transformer_type: {} as a backbone'.format(transformer_type))
        camera_num = kwargs['CAMERA_NUM']
        view_num = 0
        factory = {
            'vit_base_patch16_224_TransReID': vit_mask.vit_base_patch16_224_TransReID,
            'deit_base_patch16_224_TransReID': vit_mask.vit_base_patch16_224_TransReID,
            'vit_small_patch16_224_TransReID': vit_mask.vit_small_patch16_224_TransReID,
            'deit_small_patch16_224_TransReID': vit_mask.deit_small_patch16_224_TransReID
        }
        self.base = factory[transformer_type](img_size=kwargs['SIZE_INPUT'],
                                              sie_xishu=kwargs['SIE_COE'],
                                              local_feature=kwargs['JPM'],
                                              camera=camera_num,
                                              view=view_num,
                                              stride_size=kwargs['STRIDE_SIZE'],
                                              drop_path_rate=0.1,
                                              drop_rate=0.0,
                                              attn_drop_rate=0.0)
        if transformer_type == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_clothes = nn.Linear(self.in_planes, num_clothes, bias=False)
        self.classifier_clothes.apply(weights_init_classifier)
        self.classifier_biometric = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_biometric.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_clothes = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_clothes.bias.requires_grad_(False)
        self.bottleneck_clothes.apply(weights_init_kaiming)
        self.bottleneck_biometric = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_biometric.bias.requires_grad_(False)
        self.bottleneck_biometric.apply(weights_init_kaiming)

        # xent loss
        if xent_type == 'xent':
            self.xent = nn.CrossEntropyLoss()
            self.xent_clothes = nn.CrossEntropyLoss()
            self.xent_bio = nn.CrossEntropyLoss()
            self.loss_name = ['xent_total', 'trip_total', 'xent_bio', 'trip_bio', 'xent_clos']
        elif xent_type == 'labelsmooth':
            self.xent = CrossEntropyLabelSmooth(num_classes)
            self.xent_clothes = CrossEntropyLabelSmooth(num_classes)
            self.xent_bio = CrossEntropyLabelSmooth(num_clothes)
            self.loss_name = ['xentLS_total', 'trip_total', 'xentLS_bio', 'trip_bio', 'xentLS_clos']
        self.trip_global = TripletLoss()
        self.trip_bio = TripletLoss()
        self.trip_clothes = TripletLoss()
        self.total_loss = None
        self.loss_value = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.loss_weight = kwargs.get('loss_weight', [1.0, 1.0, 1.0, 1.0, 1.0])
        self.mae_hm_crp = nn.L1Loss()
        logging.info(f"Loss Weight of Model")
        logging.info('|'.join([i.center(15) for i in self.loss_name]))
        logging.info('|'.join([str(i).center(15) for i in self.loss_weight]))

        self.feat_weight =feat_weight
        self.use_mask = use_mask
        # mask embeding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.mask_token, std=.02)
        # normalization
        # -1 means no normalization
        self.normal_scale = normal_scale
        self.use_local_feat = kwargs.get('use_local_feat', True)

        self.use_pid_as_cloid = use_pid_as_cloid
        self.thred = thred

        self.hm_x = self.base.patch_embed.num_x
        self.hm_y = self.base.patch_embed.num_y
        self.warm_up = False

    def forward(self, img, pid, sem = None, unique_clothes_id=None, camera_id=None, view_label=None, get_hm=False):  # label is unused if self.cos_layer == 'no'
        features, patch_sem = self.base(img, sem, cam_label=camera_id, view_label=view_label)
        if unique_clothes_id is None:
            unique_clothes_id = pid
        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # decoder stage
        token = features[:, 0:1]
        img = features[:, 1:]
        img = img.permute(0, 2, 1)
        
        clos_rel_score = (patch_sem[:,:,1] > 0).float().unsqueeze(-1)
        bio_rel_score = (patch_sem[:,:,0] > 0).float().unsqueeze(-1)
        img = img.permute(0, 2, 1)
        clos_rel_score = clos_rel_score.repeat(1, 1, self.in_planes)
        bio_rel_score = bio_rel_score.repeat(1, 1, self.in_planes)
        # clothes patches decoder 
        clos_patches = torch.einsum('bpc, bpc -> bpc', img, clos_rel_score)
        if self.use_mask:
            mask_tokens = self.mask_token.repeat(clos_patches.shape[0], clos_patches.shape[1], 1)
            clos_patches = torch.where(clos_rel_score == 0, mask_tokens, clos_patches)
        local_feat_clothes_irr = self.b2(torch.cat([token, clos_patches], dim=1))
        local_feat_clothes_irr_token = local_feat_clothes_irr[:, 0]
        local_feat_clothes_token_irr_bn = self.bottleneck_clothes(local_feat_clothes_irr_token)
        
        # biometric patches decoder
        bio_patches = torch.einsum('bpc, bpc -> bpc', img, bio_rel_score)
        if self.use_mask:
            mask_tokens = self.mask_token.repeat(bio_patches.shape[0], bio_patches.shape[1], 1)
            bio_patches = torch.where(bio_rel_score == 0, mask_tokens, bio_patches)
        local_feat_bio = self.b2(torch.cat([token, bio_patches], dim=1))
        local_feat_bio_token = local_feat_bio[:, 0]
        local_feat_bio_token_bn = self.bottleneck_biometric(local_feat_bio_token)

        
        # clothes irrelevent decoder

        global_feat_bn = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(global_feat_bn)
            cls_bio = self.classifier_biometric(local_feat_bio_token_bn)
            cls_clothes = self.classifier_clothes(local_feat_clothes_token_irr_bn)
            # compute loss
            loss_xent_global = self.xent(cls_score, pid)
            loss_xent_bio = self.xent_bio(cls_bio, pid)
            loss_xent_clos = self.xent_clothes(cls_clothes, unique_clothes_id)
            loss_xent_clos = torch.exp(-loss_xent_clos)
            loss_trip_global = self.trip_global(global_feat, pid)
            loss_trip_bio = self.trip_bio(local_feat_bio_token, pid)
            #loss_trip_clos = self.trip_clothes(local_feat_clothes_token, unique_clothes_id)
            self.loss = loss_xent_global*self.loss_weight[0] + loss_trip_global*self.loss_weight[1] \
                + loss_xent_bio*self.loss_weight[2] + loss_trip_bio*self.loss_weight[3] \
                + loss_xent_clos*self.loss_weight[4]
            self.loss_value = loss_xent_global.item(), loss_trip_global.item(),\
                loss_xent_bio.item(), loss_trip_bio.item(), loss_xent_clos.item()
            self.loss_name = ['xent_total', 'trip_total', 'xent_bio', 'trip_bio', 'xent_clos']
            return cls_score, global_feat
            # global feature for triplet loss
        else:
            if self.use_local_feat:
                feat =  torch.cat([self.feat_weight[0]*global_feat, self.feat_weight[1]*local_feat_bio_token, self.feat_weight[2]*local_feat_clothes_irr_token], dim = 1)
            else:
                feat = global_feat
            
            if self.normal_scale > 0:
                feat_mod = torch.norm(feat, dim=1)
                feat = feat / feat_mod.unsqueeze(-1).repeat(1,feat.shape[1]) * self.normal_scale
            
            if get_hm:
                clos_rel_score = torch.mean(clos_rel_score, dim = 2)
                bio_rel_score = torch.mean(bio_rel_score, dim=2)
                clos_rel_score = clos_rel_score.reshape(-1, self.hm_y, self.hm_x)
                bio_rel_score = bio_rel_score.reshape(-1, self.hm_y, self.hm_x)
                return feat, [clos_rel_score, bio_rel_score]
            return feat

    def get_loss(self, *args):
        return self.loss, self.loss_value, self.loss_name
        

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))