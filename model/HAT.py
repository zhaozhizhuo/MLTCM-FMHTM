import torch
import torch.nn as nn
import random
import numpy as np

from train_parser import generate_parser
parser = generate_parser()
args = parser.parse_args()

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class HATM(nn.Module):
    def __init__(self):
        super(HATM, self).__init__()
    def forward(self, score):
        HAT_score_student = []
        HAT_score_teacher = []

        for index in range(score.shape[0]):
            sorted_alpha, indices_alpha = torch.sort(score[index,:,:], dim=-1)
            sorted_alphas = indices_alpha.tolist()
            #取前百分之二十的注意力分数作为高等级的注意力
            front_sorted_alpha = [front_sorted_alpha[:int(sorted_alpha.size(1)*0.3)] for front_sorted_alpha in sorted_alphas]
            #使用随机生成的方式来作为鲁棒性的注意力，选择20%
            random_alpha = [random.sample(range(int(sorted_alpha.size(1))), int(sorted_alpha.size(1)*0.1)) for ii in range(sorted_alpha.size(0))]
            #低于固定比例的注意力分数去除
            after_sorted_alpha = [after_sorted_alpha[:int(sorted_alpha.size(1)*0.05)] for after_sorted_alpha in sorted_alphas]

            #取高注意力和随机的并集,学生部分
            set_sorted_alpha = [set(front_sorted_alpha[i]) | set(random_alpha[i]) | set(after_sorted_alpha[i]) for i in range(len(front_sorted_alpha))]
            set_index = [1]*int(sorted_alpha.size(1))
            set_index_alpha = []
            for i in set_sorted_alpha:
                set_index_i = set_index
                for j in i:
                    set_index_i[j] = 0
                set_index_alpha.append(set_index_i)
                set_index = [1] * int(sorted_alpha.size(1))

            #取高注意力和随机的并集,教师部分
            set_index_teacher = [0] * int(sorted_alpha.size(1))
            set_index_alpha_teacher = []
            for i in front_sorted_alpha:
                set_index_i = set_index_teacher
                for j in i:
                    set_index_i[j] = 1
                set_index_alpha_teacher.append(set_index_i)
                set_index_teacher = [1] * int(sorted_alpha.size(1))


            student_score = score[index] * torch.tensor(set_index_alpha).to(score.device)
            HAT_score_student.append(student_score)
            teacher_score = score[index] * torch.tensor(set_index_alpha_teacher).to(score.device)
            HAT_score_teacher.append(teacher_score)

        HAT_score_teacher = torch.cat(HAT_score_teacher, dim=0).reshape(int(args.batch_size / len(args.gpus)),-1,score.size(1)).transpose(1,2)
        HAT_score_student = torch.cat(HAT_score_student, dim=0).reshape(int(args.batch_size / len(args.gpus)),-1,score.size(1)).transpose(1,2)

        return HAT_score_student, HAT_score_teacher