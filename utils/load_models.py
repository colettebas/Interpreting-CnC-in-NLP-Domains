from transformers import BertForSequenceClassification, BertConfig, BertTokenizerFast, pipeline
import torch

class ModelLoader():

    def __init__(self, model_checkpoint_filename):
        self.pipeline = None

        base_model = self.load_base_model()
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = self.load_model_from_checkpoint(model_checkpoint_filename, base_model)

        self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, return_all_scores=True)

    def load_base_model(self):
        config_class = BertConfig
        model_class = BertForSequenceClassification


        model_name = 'bert-base-uncased'
        num_classes = 2
        task = 'civilcomments'

        config = config_class.from_pretrained(model_name,num_labels=num_classes,finetuning_task=task)
        net = model_class.from_pretrained(model_name, from_tf=False, config=config)

        net.activation_layer = 'bert.pooler.activation'

        return net

    def load_encoder_state_dict(self, model, state_dict, contrastive_train=False):
        # Remove 'backbone' prefix for loading into model
        if contrastive_train:
            log = model.load_state_dict(state_dict, strict=False)
            for k in list(state_dict.keys()):
                print(k)
        else:
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    # Corrected for CNN
                    if k.startswith('backbone.fc1') or k.startswith('backbone.fc2'):
                        state_dict[k[len("backbone."):]] = state_dict[k]
                    # Should also be corrected for BERT models
                    elif (k.startswith('backbone.fc') or
                          k.startswith('backbone.classifier')):
                        pass
                    else:
                        state_dict[k[len("backbone."):]] = state_dict[k]
                    del state_dict[k]
            log = model.load_state_dict(state_dict, strict=False)
        print(f'log.missing_keys: {log.missing_keys}')
        return model

    def load_model_from_checkpoint(self, checkpoint_filename, base_model):
        model_state_dict = torch.load(checkpoint_filename, map_location=torch.device('cpu'))
        model_state_dict = model_state_dict['model_state_dict']
        model = self.load_encoder_state_dict(base_model, model_state_dict, contrastive_train=False)
        return model
