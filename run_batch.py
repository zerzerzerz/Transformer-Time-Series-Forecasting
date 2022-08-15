from transformer_singlestep import main, get_args, load_json

info = load_json('dataset-info.json')
args = get_args()
result_dir = 'result-vanilla-Transformer-smaller_model'
inference_method = 'fixed_len'
# inference_method = 'dynamic_decoding'
device = 'cuda:1'

for d in info:
    dataset_name = d['dataset_name']

    if dataset_name in ['ETTh1','ETTh2','ETTm1','national_illness']:
        continue

    context_len = d['context_len']
    num_channel = d['num_channel']
    for pred_len in d['pred_lens']:
        args.d_model = d['d_model']
        args.dim_feedforward = d['dim_feedforward']
        args.nhead = d['nhead']
        args.dataset_name = dataset_name
        args.context_len = context_len
        args.num_channel = num_channel
        args.pred_len = pred_len
        args.inference_method = inference_method
        args.device = device
        args.output_dir = f'{result_dir}/{dataset_name}/pred_len={pred_len}/{inference_method}'

        main(args)