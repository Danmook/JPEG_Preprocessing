import torch

def model_list(a, rank):
    if a.model_name == 'preprocess':
        from models.preprocess import Preprocess
        model = Preprocess(a, rank)
        print("Successfully load model: {}".format(a.model_name))
    else:
        if a.model_name =='preprocess_JPEG':
            from models.preprocess_JPEG import Preprocess_JPEG
            model = Preprocess_JPEG(a, rank)
            print("Successfully load model: {}".format(a.model_name))
        else:
            raise Exception('Cannot find model: {}'.format(a.model_name))
    return model

def optimizer_list(model, h):
    if h.optim_name == 'AdamW':
        optim = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
        return optim