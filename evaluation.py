# def print_metrics(metrics, suffix=None, output_path=None):
#     res = []
#     for key in ['auc_macro', 'auc_micro', 'acc_micro', 'f1_macro', 'f1_micro',
#                 'prec_at_1', 'prec_at_2', 'prec_at_3',
#                 'rec_at_1', 'rec_at_2', 'rec_at_2']:
#         res.append(metrics.get(key, '-'))
#     res = [format(r, '.4f') for r in res]
#     if output_path is None:
#         print('------')
#         if suffix is not None:
#             print(suffix)
#         print('MACRO-auc, MICRO-auc, ACC, MACRO-f1, MICRO-f1, P@1, P@2, P@3, R@1, R@2, R@3')
#         print(', '.join(res))
#     else:
#         with open(output_path, "a", encoding="utf-8") as f:
#             f.write('------\n')
#             if suffix is not None:
#                 f.write(f'{suffix}\n')
#             f.write('MACRO-auc, MICRO-auc, ACC, MACRO-f1, MICRO-f1, P@1, P@2, P@3, R@1, R@2, R@3\n')
#             f.write(', '.join(res) + '\n')
#
#     # main icd metric ooutput
#     res = []
#     if 'main_label_acc' in metrics:
#         for key in ['main_label_acc', 'h@5', 'h@8', 'h@15', 'mrr']:
#             res.append(metrics.get(key, '-'))
#         res = [format(r, '.4f') for r in res]
#         if output_path is None:
#             print('------')
#             if suffix is not None:
#                 print(suffix)
#             print('main_label_acc, h@5, h@8, h@15, mrr')
#             print(', '.join(res))
#         else:
#             with open(output_path, "a", encoding="utf-8") as f:
#                 f.write('------\n')
#                 if suffix is not None:
#                     f.write(f'{suffix}\n')
#                 f.write('main_label_acc, h@5, h@8, h@15, mrr\n')
#                 f.write(', '.join(res) + '\n')
#

def print_metrics(metrics, suffix=None, output_path=None):
    res = []
    for key in ['auc_macro', 'auc_micro', 'acc_macro', 'acc_micro', 'f1_macro', 'f1_micro',
                'prec_at_1', 'prec_at_2', 'prec_at_3',
                'rec_at_1', 'rec_at_2', 'rec_at_2']:
        res.append(metrics.get(key, '-'))
    res = [format(r, '.4f') for r in res]
    if output_path is None:
        print('------')
        if suffix is not None:
            print(suffix)
        print('MACRO-auc, MICRO-auc, MACRO-ACC, MICEO-ACC, MACRO-f1, MICRO-f1, P@1, P@2, P@3, R@1, R@2, R@3')
        print(', '.join(res))
    else:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write('------\n')
            if suffix is not None:
                f.write(f'{suffix}\n')
            f.write('MACRO-auc, MICRO-auc, MACRO-ACC, MICEO-ACC, MACRO-f1, MICRO-f1, P@1, P@2, P@3, R@1, R@2, R@3\n')
            f.write(', '.join(res) + '\n')

    # main icd metric ooutput
    res = []
    if 'main_label_acc' in metrics:
        for key in ['main_label_acc', 'h@5', 'h@8', 'h@15', 'mrr']:
            res.append(metrics.get(key, '-'))
        res = [format(r, '.4f') for r in res]
        if output_path is None:
            print('------')
            if suffix is not None:
                print(suffix)
            print('main_label_acc, h@5, h@8, h@15, mrr')
            print(', '.join(res))
        else:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write('------\n')
                if suffix is not None:
                    f.write(f'{suffix}\n')
                f.write('main_label_acc, h@5, h@8, h@15, mrr\n')
                f.write(', '.join(res) + '\n')

