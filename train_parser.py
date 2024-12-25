import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='../data_preprocess', help="batch size")   #146分类 ../data_preprocess   #pure分类 ../data_preprocess/pure_mul

    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learn rate")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")
    parser.add_argument("--gpus", type=list, default=[0], help="gpus")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--neg_number", type=int, default=4, help="neg_number")

    parser.add_argument("--syndrome_diag", type=str, default='syndrome', help="syndrome or diagnosis")
    parser.add_argument("--model_path", type=str, default='chinese_wwm_pytorch', help="model_path")


    #prompt length
    # parser.add_argument("--prompt_length", type=int, default=200, help="prompt_length")
    parser.add_argument("--max_length", type=int, default=200, help="max_length")
    parser.add_argument("--define_length", type=int, default=500, help="define_length")
    parser.add_argument("--for_num", type=int, default=1, help="for_num")
    parser.add_argument("--divider", type=bool, default=True, help="divider")

    # lstm
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden")

    #laat
    parser.add_argument("--d_a", type=int, default=256, help="d_a")
    parser.add_argument("--n_labels", type=int, default=146, help="n_labels")  #zybert 146  szy 7  pure 46

    #ham
    parser.add_argument("--rand_ham", type=int, default=100, help="rand_ham")

    parser.add_argument("--big", type=str, default='True', help="dataset") # True small Flase

    parser.add_argument("--model", type=str, default='longformer', help="model")  #bert / longformer
    parser.add_argument("--threshold", type=float, default=None, help="threshold")
    parser.add_argument("--prob_threshold", type=float, default=0, help="threshold")

    parser.add_argument("--dataset", type=str, default='tcm_szy', help="dataset")  #tcm_szy  #tcm-sd

    return parser
