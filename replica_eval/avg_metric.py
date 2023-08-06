import os
import numpy as np


root_dir = './evaluation_results' # path to the evaluation results folder

all_metric = []
for fname in os.listdir(root_dir):
    metric_result = np.load(os.path.join(root_dir, fname, 'metrics_3D_obj.npy'))
    if np.isnan(metric_result.any()):
        print(fname)
    # print(metric_result.shape)
    all_metric.append(metric_result)
    print(fname, 0.5*(metric_result.mean(1)[0] + metric_result.mean(1)[1]))
result = np.concatenate(all_metric, 1)
# # recompute the f-score from precision and recall
precision_1, completion_1 = result[4], result[2]
_sum = precision_1 + completion_1
_prod = 2*precision_1 * completion_1


print('var F_score 5cm: {}'.format(result.var(1)[7]))
print('var Chamer: {}'.format((0.5*(result[0]+result[1])).var(0)))
# print('chamfer mean each scene: {}'.format(0.5*(result.mean(1)+result.mean(2))))
print('Acc mean: {}, Comp: {}, chamfer: {}, Ratio 1cm: {}, Ratio 5cm: {}, F_score 1cm: {}, F_score 5cm: {}'.format(result.mean(1)[0], result.mean(1)[1], 0.5*(result.mean(1)[0]+result.mean(1)[1]), result.mean(1)[2], result.mean(1)[3], result.mean(1)[6], result.mean(1)[7]))
