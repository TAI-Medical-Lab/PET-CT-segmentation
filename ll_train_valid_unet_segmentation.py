import torch
import torch.nn as nn
from utils import computeDiceOneHot, getOneHotSegmentation, getTargetSegmentation, predToSegmentation, DicesToDice
import torch.distributed as dist
from config_segmentation import config
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.functional import cosine_similarity
import torch.distributed as dist
import torch.nn.functional as F
import copy

opx = config()
opt = opx()


def cosine_similarity_weighted_average(model):
    world_size = 4
    
    # Average model weights across all processes
    avg_all_params = []
    for param in model.parameters():
        avg_param = param.data.clone()
        dist.all_reduce(avg_param, op=dist.ReduceOp.SUM)
        avg_param /= world_size
        avg_all_params.append(avg_param)

    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])
    
    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)
    
    # Compute average of all flattened parameters
    avg_all_flat_params = torch.cat([param.view(-1) for param in avg_all_params])

    # Calculate cosine similarity between the average and all processes
    similarities = torch.tensor([F.cosine_similarity(avg_all_flat_params, other_flat_params, dim=-1) for other_flat_params in all_flat_params_list])
    
    # if opt.re_cosine == 1:
    #     similarities = 1 - similarities

    softmax_weights = F.softmax(similarities, dim=0)
    

    # Calculate softmax weights

    # Compute the weighted average of parameters
    # weighted_avg_params = torch.stack([weight * params for weight, params in zip(softmax_weights, all_flat_params_list)], dim=0).sum(dim=0)
    
    # Update the model with the weighted average parameters
    # index = 0
    # for param in model.parameters():
    #     numel = param.numel()
    #     param.data.copy_(weighted_avg_params[index:index + numel].view(param.shape))
    #     index += numel
    # print(softmax_weights)
    return softmax_weights


def coordinate_wise_median(model):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Average model weights across all processes
    avg_all_params = []
    for param in model.parameters():
        avg_param = param.data.clone()
        dist.all_reduce(avg_param, op=dist.ReduceOp.SUM)
        avg_param /= world_size
        avg_all_params.append(avg_param)

    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])

    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)

    # Flatten the average parameters
    avg_all_flat_params = torch.cat([param.view(-1) for param in avg_all_params])

    # Calculate the absolute difference between each local model parameter and the average
    abs_diff_list = [torch.abs(local_flat_params - avg_all_flat_params) for local_flat_params in all_flat_params_list]

    # Sum the absolute differences from all processes using all_reduce
    sum_abs_diff_list = [abs_diff.clone() for abs_diff in abs_diff_list]
    for sum_abs_diff in sum_abs_diff_list:
        dist.all_reduce(sum_abs_diff, op=dist.ReduceOp.SUM)
        sum_abs_diff /= world_size

    print(sum_abs_diff)

    # Calculate the median of the summed absolute differences
    median_abs_diff_list = [torch.median(sum_abs_diff) for sum_abs_diff in sum_abs_diff_list]

    # Update the model parameter with the median
    index = 0
    for param, median_abs_diff in zip(model.parameters(), median_abs_diff_list):
        numel = param.numel()
        param.data.copy_(avg_all_params[index] - median_abs_diff)
        index += 1



def inver_ratio_weighted_average(model):
    world_size = 4
    
    # Average model weights across all processes
    avg_all_params = []
    for param in model.parameters():
        if opt.DP == 'yes':
            param.data = add_gaussian(param.data, delta=opt.dp_delta)

        avg_param = param.data.clone()
        dist.all_reduce(avg_param, op=dist.ReduceOp.SUM)
        avg_param /= world_size
        avg_all_params.append(avg_param)

    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])
    
    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)
    
    # Compute average of all flattened parameters
    avg_all_flat_params = torch.cat([param.view(-1) for param in avg_all_params])

    # Calculate L2 norm of average parameters
    avg_norm = torch.norm(avg_all_flat_params)

    # Calculate L2 norm of each process's parameters and ratio to the average parameters
    # norms = [torch.norm(param) for param in model.parameters()]
    # ratios = [avg_norm / norm for norm in norms]
    # Calculate L2 norm of each process's flattened parameters
    norms = [torch.norm(flat_params) for flat_params in all_flat_params_list]
    ratios = [avg_norm / norm for norm in norms]


    # Apply softmax to the ratios to obtain weights
    ratios_tensor = torch.tensor(ratios)

    inverse_ratios_tensor = 1 / ratios_tensor
    weights = F.softmax(inverse_ratios_tensor, dim=0).tolist()
    
    # Compute the weighted average of parameters
    # weighted_avg_params = torch.stack([weight * params for weight, params in zip(weights, all_flat_params_list)], dim=0).sum(dim=0)

    # # Update the model with the weighted average parameters
    # index = 0
    # for param in model.parameters():
    #     numel = param.numel()
    #     param.data.copy_(weighted_avg_params[index:index + numel].view(param.shape))
    #     index += numel
    print(weights)
    return weights

def inver_pccs_weighted_average(model):
    world_size = 4
    
    # Average model weights across all processes
    avg_all_params = []
    for param in model.parameters():
        avg_param = param.data.clone()
        dist.all_reduce(avg_param, op=dist.ReduceOp.SUM)
        avg_param /= world_size
        avg_all_params.append(avg_param)

    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])
    
    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)
    
    # Compute average of all flattened parameters
    avg_all_flat_params = torch.cat([param.view(-1) for param in avg_all_params])

    # Calculate pccs similarity between the average and all processes
    # similarities = torch.tensor([torch.nn.functional.linear(avg_all_flat_params.unsqueeze(0), other_flat_params.unsqueeze(1)).squeeze() for other_flat_params in all_flat_params_list])
    similarities = torch.tensor([torch.dot(avg_all_flat_params, other_flat_params) for other_flat_params in all_flat_params_list])
    
    # print(similarities)
    # if opt.re_pccs == 0:
    similarities = 1 / similarities
        
    softmax_weights = F.softmax(similarities, dim=0)
    # print(softmax_weights)
    return softmax_weights


def inver_cosine_similarity_weighted_average(model):
    world_size = 4
    
    # Average model weights across all processes
    avg_all_params = []
    for param in model.parameters():
        avg_param = param.data.clone()
        dist.all_reduce(avg_param, op=dist.ReduceOp.SUM)
        avg_param /= world_size
        avg_all_params.append(avg_param)

    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])
    
    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)
    
    # Compute average of all flattened parameters
    avg_all_flat_params = torch.cat([param.view(-1) for param in avg_all_params])

    # Calculate cosine similarity between the average and all processes
    similarities = torch.tensor([F.cosine_similarity(avg_all_flat_params, other_flat_params, dim=-1) for other_flat_params in all_flat_params_list])
    
    # Calculate inverse of cosine similarities
    inverse_similarities = 1 / similarities

    # Calculate softmax weights
    softmax_weights = F.softmax(inverse_similarities, dim=0)
    
    # Compute the weighted average of parameters
    # weighted_avg_params = torch.stack([weight * params for weight, params in zip(softmax_weights, all_flat_params_list)], dim=0).sum(dim=0)
    
    # # Update the model with the weighted average parameters
    # index = 0
    # for param in model.parameters():
    #     numel = param.numel()
    #     param.data.copy_(weighted_avg_params[index:index + numel].view(param.shape))
    #     index += numel
    print(softmax_weights)
    return softmax_weights

def add_gaussian(updates,delta=0.001):
    '''inject gaussian noise to a vector'''
    updates += torch.FloatTensor(np.random.normal(0, delta,updates.shape)).to(torch.device("cuda"))
    return updates

def average_weights(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        grad_tensor = param.data

        if opt.DP == 'yes':
            grad_tensor = param.data
            grad_tensor = add_gaussian(grad_tensor, delta=opt.dp_delta)
        
        dist.all_reduce(grad_tensor,op=dist.ReduceOp.SUM)
        param.data /= size

def norm_params(model):
    # Get flattened gradients for current model
    flat_params = torch.cat([param.grad.view(-1) for param in model.parameters()])

    world_size = 4

    # Get flattened gradients from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)

    # Compute the L2 norm of flattened gradients from all processes
    squared_l2_norms = torch.tensor([torch.sum(torch.square(param)) for param in all_flat_params_list])
    l2_norms = torch.sqrt(squared_l2_norms)
    # Normalize L2 norms
    l2_norms_normalized = l2_norms / torch.sum(l2_norms)

    return l2_norms_normalized

def qfedavg_weight(optimizer, loss, norm, max_loss):
    q = opt.q
    lr = optimizer.param_groups[0]['lr']
    normalized_loss = torch.tensor(loss) / torch.tensor(max_loss)
    return (q * torch.float_power(normalized_loss + 1e-10, (q - 1)) * torch.tensor(norm) + (1.0 / lr) * torch.float_power(normalized_loss + 1e-10, q)).item()

def qfedavg_aggregate_weights(model, weights):
    world_size = 4
    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])
    
    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)
    

    # Compute the weighted average of parameters
    weighted_avg_params = torch.stack([weight * params for weight, params in zip(weights, all_flat_params_list)], dim=0).sum(dim=0)
    
    # Update the model with the weighted average parameters
    index = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(weighted_avg_params[index:index + numel].view(param.shape))
        index += numel

def qfedavgs(model, optimizer, loss):
    weights = [0.0 for _ in range(4)]
    losses = [0.0 for _ in range(4)]

    l2_norms = norm_params(model)
    norm = l2_norms[dist.get_rank()].item()
    loss = loss.item()
    losses[dist.get_rank()] = loss

    # Gather all losses
    all_losses_tensor = torch.tensor(losses)
    gathered_losses = [torch.zeros_like(all_losses_tensor) for _ in range(4)]
    dist.all_gather(gathered_losses, all_losses_tensor)

    # Convert gathered_losses to a flat list of scalar values
    gathered_loss_values = [loss.item() for gathered_loss in gathered_losses for loss in gathered_loss]

    # Calculate the maximum loss across all processes
    max_loss_tensor = torch.tensor(max(gathered_loss_values))
    max_loss_tensor_list = [torch.zeros_like(max_loss_tensor) for _ in range(4)]
    dist.all_gather(max_loss_tensor_list, max_loss_tensor)
    max_loss = max([x.item() for x in max_loss_tensor_list])

    weight = qfedavg_weight(optimizer, loss, norm, max_loss)
    weights[dist.get_rank()] = weight

    # 将4个进程对应的长度为4的weights，汇总成1个长度为4的权重数组，其中最后的权重数组对应的权重分别对应着进程号 0 1 2 3
    # 汇总权重数组
    all_weights_tensor = torch.tensor(weights)
    gathered_weights = [torch.zeros_like(all_weights_tensor) for _ in range(4)]
    dist.all_gather(gathered_weights, all_weights_tensor)

    # 将4个进程对应的长度为4的weights，汇总成1个长度为4的权重数组
    weights = [0.0 for _ in range(4)]
    for i, gathered_weight in enumerate(gathered_weights):
        weights[i] = gathered_weight[i]

    # 计算所有进程对应的权重之和
    weight = weight / sum(weights)

    weight1 = torch.tensor(weight)
    weight2 = [torch.zeros_like(weight1) for _ in range(4)]
    dist.all_gather(weight2, weight1)

    # print(weight2)
    return weight2

def CGSV(model, reputations, alpha=0.9):
    world_size = len(reputations)

    # 1. Normalize gradients for the current model
    flat_params = torch.cat([param.grad.view(-1) for param in model.parameters()])
    grad_norm = flat_params.norm(2)
    normalized_grad = flat_params / grad_norm

    # 2. Gather normalized gradients from all clients
    all_flat_params_list = [torch.zeros_like(normalized_grad) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, normalized_grad)

    # 3. Aggregate normalized gradients using reputation as weights
    weighted_avg_params = torch.stack([reputation * params for reputation, params in zip(reputations, all_flat_params_list)], dim=0).sum(dim=0)

    # 4. Calculate cosine similarity between normalized gradients and aggregated gradients
    cosine_sim = cosine_similarity(normalized_grad, weighted_avg_params, dim=0).item()

    # 5. Update the reputation value of the current client using EWMA
    rank = dist.get_rank()
    reputations[rank] = alpha * reputations[rank] + (1 - alpha) * cosine_sim

    # 6. Calculate Q-values
    reputations_sum = sum(reputations)
    Q_values = [reputation / reputations_sum for reputation in reputations]

    # print(Q_values)
    return Q_values

def compensate_minimum_weight(weights, min_threshold=opt.min_threshold):
    num_clients = len(weights)
    sum_weight = sum(weights)
    
    min_weight = min(weights)
    index_min = weights.index(min_weight)
    
    if min_weight < min_threshold:
        compensation = (min_threshold - min_weight) / (num_clients - 1)
        
        for i in range(num_clients):
            if i != index_min:
                weights[i] += compensation
        
        weights[index_min] = min_threshold

    # Make sure the weights still sum up to 1 (or very close to 1 due to floating point precision)
    #assert abs(sum(weights) - sum_weight) < 1e-8, "The sum of the weights changed after compensation"
    
    return weights

def train(model, crierion, data_loader_train, optimizer, epoch):
    model.train()
    loss_train = 0.
    DiceB_train = 0.
    voe_score_train=0.
    # DiceW_train = 0.
    # DiceT_train = 0.
    # DiceZ_train = 0.
    Dice_score_train = 0.
    softmax = nn.Softmax()
    Diceloss = computeDiceOneHot()
    softmax.to(torch.device("cuda"))
    Diceloss.to(torch.device("cuda"))
    
    i = 0
    for batch_idx,(ct_image, pet_image, label) in tqdm(enumerate(data_loader_train), total=len(data_loader_train)):
        i = i + 1
        if i != len(data_loader_train):
            ct_image, pet_image, label = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")),\
                                        label.to(torch.device("cuda"))

            optimizer.zero_grad()
            # with autocast(flag):
            segmentation_prediction = model(ct_image, pet_image)
            predclass_y = softmax(segmentation_prediction)
            Segmentation_planes = getOneHotSegmentation(label)
            segmentation_prediction_ones = predToSegmentation(predclass_y)
            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(label)
            # print(segmentation_prediction.shape)
            # print(Segmentation_class.shape)
            loss = crierion(segmentation_prediction, Segmentation_class)
            loss_train += loss.item()
            # 计算DSC
            # 分5类
            # DiceN, DiceB, DiceW, DiceT, DiceZ = Diceloss(segmentation_prediction_ones, Segmentation_planes)
            # 分2类
            DiceN, DiceB ,voe= Diceloss(segmentation_prediction_ones, Segmentation_planes)
            DiceB = DicesToDice(DiceB)
            # DiceW = DicesToDice(DiceW)
            # DiceT = DicesToDice(DiceT)
            # DiceZ = DicesToDice(DiceZ)
            # 分5类
            # Dice_score = (DiceB + DiceW + DiceT + DiceZ) / 4
            # 分2类
            # Dice_score = (DiceB + DiceW) / 2
            Dice_score = (DiceB) / 2
            DiceB_train += DiceB.item()
            # DiceW_train += DiceW.item()
            # DiceT_train += DiceT.item()
            # DiceZ_train += DiceZ.item()
            Dice_score_train += Dice_score.item()

            voe_score_train+=voe.item()

            loss.backward()
            optimizer.step()
            # scale.scale(loss).backward()
            # scale.step(optimizer)
            # scale.update()
            
            if opt.setting == 'FL':
                average_weights(model)

            elif opt.setting=='comed':
                coordinate_wise_median(model)

            elif opt.setting == 'FL_re_cosine':  
                weight = cosine_similarity_weighted_average(model)
                # weight = inver_cosine_similarity_weighted_average(model)
                # 加权聚合
                qfedavg_aggregate_weights(model, weight)
                
            elif opt.setting == 'FL_re_distance':  
                weight = inver_ratio_weighted_average(model)
                qfedavg_aggregate_weights(model, weight)

            elif opt.setting == 'QFFedAvg':
                weight = qfedavgs(model, optimizer,loss,0.5)
                qfedavg_aggregate_weights(model, weight)

            elif opt.setting == 'CGSV':
                reputations = [1.0, 1.0, 1.0, 1.0]
                weight = CGSV(model, reputations, alpha=0.9)
                qfedavg_aggregate_weights(model, weight)

            elif (opt.setting == 'ours' and epoch<100 and opt.DP=='yes') or (opt.setting == 'ours' and opt.DP=='no'):
                # # 是否增加微量噪声
                # if opt.DP == 'yes':
                #     for param in model.parameters():
                #         grad_tensor = param.data
                #         grad_tensor = add_gaussian(grad_tensor, delta=opt.dp_delta)
                #
                #         dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
                weight = [0.25 for _ in range(4)]
                weight_Dir = [0.25 for _ in range(4)]
                weight_Dis = [0.25 for _ in range(4)]
                weight_loss = [0.25 for _ in range(4)]
                weight_pccs = [0.25 for _ in range(4)]
                if opt.ours_dir == 1:
                    weight_Dir = cosine_similarity_weighted_average(model)
                    print('weight_Dir')
                    print(weight_Dir)

                if opt.ours_dis == 1:
                    weight_Dis = inver_ratio_weighted_average(model)
                
                if opt.ours_loss == 1:
                    print('weight_loss')
                    weight_loss = qfedavgs(model, optimizer,loss)
                    print(weight_loss)

                if opt.ours_pccs == 1:
                    weight_pccs = inver_pccs_weighted_average(model)
                    print('weight_pccs')
                    print(weight_pccs)

                # 定义 lambda 权重
                lambda_weight = opt.weight  # 例如，可以将 lambda 设置为 0.25
                
                # 遍历所有可能的权重组合
                if opt.ours_dir + opt.ours_loss + opt.ours_dis + opt.ours_pccs == '4':
                    weight_combinations = [
                        [weight_Dir, weight_Dis, weight_pccs, weight_loss]
                    ]

                    
                    # 定义 lambda 权重
                    # lambda_weight = opt.weight  # 例如，可以将 lambda 设置为 0.25

                    
                    # 平均权重计算
                    non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:3])]
                    loss_weight_combinations = weight_combinations[3]
                    
                    # 计算加权平均
                    weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                    weight = weight_combinations


                if opt.ours_dir + opt.ours_loss + opt.ours_dis + opt.ours_pccs == 3:
                    if opt.ours_dir == 1 and opt.ours_loss == 1 and opt.ours_dis == 1:
                            weight_combinations = [
                                [weight_Dir, weight_Dis, weight_loss]
                            ]
                           

                            # 平均权重计算
                            non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:2])]
                            loss_weight_combinations = weight_combinations[2]
                            
                            # 计算加权平均
                            weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                            weight = weight_combinations

                    elif opt.ours_dir == 1 and opt.ours_loss == 1 and opt.ours_pccs == 1:
                        weight_combinations = [
                            [weight_Dir, weight_pccs, weight_loss]
                        ]
                        # 定义 lambda 权重
                        # lambda_weight = opt.weight  # 例如，可以将 lambda 设置为 0.25

                        # 平均权重计算
                        # non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:2])]
                        # loss_weight_combinations = weight_combinations[2]
                       
                       # 平均权重计算
                        non_loss_weight_combinations = [sum(w) / len(w) for w in zip(weight_Dir, weight_pccs)]
                        loss_weight_combinations = weight_loss

                        # 计算加权平均
                        weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                        weight = weight_combinations

                        # weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        # weight = weight_combinations
                    elif opt.ours_dir == 1 and opt.ours_dis == 1 and opt.ours_pccs == 1:
                        weight_combinations = [
                            [weight_Dir, weight_Dis, weight_pccs]
                        ]
                        # 平均权重计算
                        weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        weight = weight_combinations
                    
                    elif opt.ours_loss == 1 and opt.ours_dis == 1 and opt.ours_pccs == 1:
                        weight_combinations = [
                            [weight_Dis, weight_pccs,weight_loss]
                        ]
                        # 定义 lambda 权重
                        lambda_weight = opt.weight  # 例如，可以将 lambda 设置为 0.25

                        # 平均权重计算
                        non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:2])]
                        loss_weight_combinations = weight_combinations[2]
                        
                        # 计算加权平均
                        weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                        weight = weight_combinations

                        # 平均权重计算
                        # weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        # weight = weight_combinations


                if opt.ours_dir + opt.ours_loss + opt.ours_dis + opt.ours_pccs == 2:
                    if opt.ours_dir == 1 and opt.ours_loss == 1:
                        weight_combinations = [
                            [weight_Dir, weight_loss]
                        ]
                        
                        # 平均权重计算
                        non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:1])]
                        loss_weight_combinations = weight_combinations[1]
                        
                        # 计算加权平均
                        weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                        weight = weight_combinations
                        
                        # weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        # weight = weight_combinations
                    elif opt.ours_dir == 1 and opt.ours_dis == 1:
                        weight_combinations = [
                            [weight_Dir, weight_Dis]
                        ]
                        weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        weight = weight_combinations
                    elif opt.ours_dir == 1 and opt.ours_pccs == 1:
                        weight_combinations = [
                            [weight_Dir, weight_pccs]
                        ]
                        weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        weight = weight_combinations
                    elif opt.ours_loss == 1 and opt.ours_dis == 1:
                        weight_combinations = [
                            [weight_Dis, weight_loss]
                        ]
                        # 平均权重计算
                        non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:1])]
                        loss_weight_combinations = weight_combinations[1]
                        
                        # 计算加权平均
                        weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                        weight = weight_combinations
                    elif opt.ours_loss == 1 and opt.ours_pccs == 1:
                        weight_combinations = [
                            [weight_pccs, weight_loss]
                        ]
                        # 平均权重计算
                        non_loss_weight_combinations = [sum(w)/len(w) for w in zip(*weight_combinations[:1])]
                        loss_weight_combinations = weight_combinations[1]
                        
                        # 计算加权平均
                        weight_combinations = [((1 - lambda_weight) * w1 + lambda_weight * w2) for w1, w2 in zip(non_loss_weight_combinations, loss_weight_combinations)]

                        weight = weight_combinations
                        
                        # weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        # weight = weight_combinations
                    elif opt.ours_dis == 1 and opt.ours_pccs == 1:
                        weight_combinations = [
                            [weight_Dis, weight_pccs]
                        ]
                        weight_combinations = [[(a + b) / 2 for a, b in zip(weights[0], weights[1])] for weights in weight_combinations]
                        weight = weight_combinations
                                
                # low bound 最低权重补偿
                print(weight)
                if opt.low_bound_ensure == 1:
                    weight = compensate_minimum_weight(weight_loss, min_threshold=opt.min_threshold)


                print('final_weight')
                print(weight)
                # 加权聚合
                qfedavg_aggregate_weights(model, weight)
                # # 是否增加微量噪声
                # if opt.DP == 'yes' :
                #     for param in model.parameters():
                #         grad_tensor = param.data
                #         grad_tensor = add_gaussian(grad_tensor, delta=opt.dp_delta)
                #
                #         dist.all_reduce(grad_tensor,op=dist.ReduceOp.SUM)

            elif opt.setting=='ours' and opt.DP=='yes' and epoch>=100:
                average_weights(model)



            # 2分类
            if (batch_idx+1)/(len(data_loader_train)) == 1:
                print('Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f},voe:{:.6f}'.format(
                    epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
                    DiceB_train/len(data_loader_train),voe_score_train/len(data_loader_train)))
    return 'Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f}'.format(
                    epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
                    DiceB_train/len(data_loader_train))


def valid(model, criterion, data_loader_valid, epoch):
    model.eval()
    softmax = nn.Softmax()
    Diceloss = computeDiceOneHot()
    softmax.to(torch.device("cuda"))
    Diceloss.to(torch.device("cuda"))
    dice = 0.
    dice1 = 0.
    dice1_mean = 0.
    voe_score_valid=0.
    with torch.no_grad():
        for batch_idx, (ct_image, pet_image, target) in enumerate(data_loader_valid):
            ct_image, pet_image, target = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")), \
                                         target.to(torch.device("cuda"))
            segmentation_prediction = model(ct_image, pet_image)
            pred_y = softmax(segmentation_prediction)
            Segmentation_planes = getOneHotSegmentation(target)
            segmentation_prediction_ones = predToSegmentation(pred_y)
            DicesN, Dices1 ,voe= Diceloss(segmentation_prediction_ones, Segmentation_planes)
            Dice1 = DicesToDice(Dices1)
            dice_score = (Dice1) / 2
            dice += dice_score.item()
            dice1 += Dice1.item()
            voe_score_valid+=voe.item()


    # 等待所有进程完成计算
    dist.barrier()

    # 使用分布式数据并行计算模型在验证集上的结果
    dice1_tensor = torch.tensor((dist.get_rank(), dice1 / len(data_loader_valid), dice1_mean / len(data_loader_valid), voe_score_valid/len(data_loader_valid))).to(torch.device("cuda"))
    gathered_dice1 = [torch.zeros_like(dice1_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_dice1, dice1_tensor)
    dice1_list = [(d[0].item(), d[1].item(), d[2].item(), d[3].item()) for d in gathered_dice1]

    # 计算所有进程的dice1平均值和dice1_mean的平均值
    dice1_avg = sum([d[1] for d in dice1_list]) / len(dice1_list)
    dice1_mean_avg = sum([d[2] for d in dice1_list]) / len(dice1_list)
    voe_avg = sum([d[3] for d in dice1_list]) / len(dice1_list)

    # 输出所有进程的dice和dice1平均值
    all_rank_dice1 = [d[1] for d in dice1_list]
    all_rank_voe= [d[3] for d in dice1_list]
    
    
    print('dice1_list',dice1_list)

    print('Valid Epoch: {}, All_Rank_DICE1: {}'.format(epoch, all_rank_dice1))
    print('Valid Epoch: {}, All_Rank_Avg_DICE1: {:.6f}'.format(epoch, dice1_avg))
    print('Valid Epoch: {}, All_Rank_Mean_DICE: {:.6f}'.format(epoch, dice1_mean_avg))

    return epoch, all_rank_dice1,dice1_mean_avg, dice1_avg, all_rank_voe,voe_avg

    

def valid2(model, crierion, data_loader_valid, epoch):
    model.eval()
    softmax = nn.Softmax()
    Diceloss = computeDiceOneHot()
    softmax.to(torch.device("cuda"))
    Diceloss.to(torch.device("cuda"))
    dice = 0.
    dice1 = 0.
    voe_score_valid=0.
    # dice2 = 0.
    # dice3 = 0.
    # dice4 = 0.
    with torch.no_grad():
        for batch_idx, (ct_image, pet_image, target) in enumerate(data_loader_valid):
            ct_image, pet_image, target = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")), \
                                         target.to(torch.device("cuda"))
            segmentation_prediction = model(ct_image, pet_image)
            pred_y = softmax(segmentation_prediction)
            Segmentation_planes = getOneHotSegmentation(target)
            segmentation_prediction_ones = predToSegmentation(pred_y)
            # 分5类
            # DicesN, Dices1, Dices2, Dices3, Dices4 = Diceloss(segmentation_prediction_ones, Segmentation_planes)
            # 分2类
            DicesN, Dices1,voe = Diceloss(segmentation_prediction_ones, Segmentation_planes)
            Dice1 = DicesToDice(Dices1)
            # Dice2 = DicesToDice(Dices2)
            # Dice3 = DicesToDice(Dices3)
            # Dice4 = DicesToDice(Dices4)
            # 分5类
            # dice_score = (Dice1 + Dice2 + Dice3 + Dice4) / 4
            # 分2类
            # dice_score = (Dice1 + Dice2) / 2
            # dice += dice_score.item()
            # dice1 += Dice1.item()
            # dice2 += Dice2.item()
            # 2分类
            dice_score = (Dice1) / 2
            dice += dice_score.item()
            dice1 += Dice1.item()
            voe_score_valid+=voe.item()
            # dice3 += Dice3.item()
            # dice4 += Dice4.item()
            # 分5类
            # if (batch_idx + 1) / (len(data_loader_valid)) == 1:
            #     print('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f}, Dice2: {:.6f}, Dice3: {:.6f}, Dice4: {:.6f}'.format(
            #         epoch, dice / len(data_loader_valid), dice1 / len(data_loader_valid),
            #         dice2 / len(data_loader_valid), dice3 / len(data_loader_valid), dice4 / len(data_loader_valid)))
            # 分2类
            # if (batch_idx + 1) / (len(data_loader_valid)) == 1:
            #     print('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f}, Dice2: {:.6f}'.format(
            #         epoch, dice / len(data_loader_valid), dice1 / len(data_loader_valid),
            #         dice2 / len(data_loader_valid)))
            # 2分类
            if (batch_idx + 1) / (len(data_loader_valid)) == 1:
                print('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f}, voe: {:.6f}'.format(
                    epoch, dice / len(data_loader_valid), dice1 / len(data_loader_valid),voe_score_valid / len(data_loader_valid)))

    return epoch, dice/len(data_loader_valid), dice1/len(data_loader_valid),voe_score_valid/len(data_loader_valid),


