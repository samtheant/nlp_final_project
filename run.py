import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser, PreTrainedModel, \
    PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import pandas as pd
import torch

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True, default='nli',
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--hypothesis_only', type=bool, default=None,
                      help='Hypothesis only training')
    argp.add_argument('--hybrid', type=bool, default=None,
                      help='Hybrid model')
    argp.add_argument('--biased_model_type', type=str, default=None,
                      help='Type of biased model to train {hypothesis_only, sequence_length, both}')

    training_args, args = argp.parse_args_into_dataclasses()

    training_args.num_train_epochs = 1

    # Dataset selection
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        # dataset = datasets.load_dataset('json', data_files=args.dataset)
        df = pd.read_csv('contrast_set.csv')
        dataset = datasets.Dataset.from_pandas(df)

        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        # eval_split = 'train'
        eval_split = 'test'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    def remove_premise(input_tensor):
        output_tensor = torch.zeros_like(input_tensor)

        for i in range(input_tensor.size(0)):
            row = input_tensor[i]
            index = (row == 102).nonzero()[0]
            output_tensor[i, 0] = 101
            output_tensor[i, 1:len(row[index+1:])+1] = row[index+1:]
            
        return output_tensor

    class SentenceLengthModel(torch.nn.Module):
        def __init__(self):
            super(SentenceLengthModel, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(2, 50), 
                torch.nn.ReLU(), 
                torch.nn.Linear(50, 3)
            )
        def forward(self, input_ids = None,
                        attention_mask = None,
                        token_type_ids = None,
                        position_ids = None,
                        head_mask = None,
                        inputs_embeds = None,
                        labels = None,
                        output_attentions = None,
                        output_hidden_states = None,
                        return_dict = None,):
            seps = (input_ids == 102).nonzero(as_tuple=False)
            seps = seps[:, 1].view(-1, 2)
            seps[:, 1] = seps[:, 1] - seps[:, 0]
            seps = (seps - 7) / 15
            preds = self.model(seps.float())
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(preds, labels)
            return SequenceClassifierOutput(
                        loss = loss, 
                        logits = preds
                    )

          

    if args.hybrid == True:
        class MyConfig(PretrainedConfig):
            model_type = 'hybrid'
            def __init__(self, important_param=42, **kwargs):
                super().__init__(**kwargs)
                self.important_param = important_param
                self.num_labels = 3
        class Hybrid(PreTrainedModel):
            config_class = MyConfig
            def __init__(self, config):
                super(Hybrid, self).__init__(config)
                self.crap_model1 = None
                self.crap_model2 = None
                self.crap_model = None
                self.regular_model = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', **task_kwargs)
                if args.biased_model_type == 'sequence_length' or args.biased_model_type == 'both':
                    self.crap_model1 = SentenceLengthModel()
                    self.crap_model1.load_state_dict(torch.load("./sequence_length_biased_model/pytorch_model.bin"))
                    for name, param in self.crap_model1.named_parameters():
                        param.requires_grad = False
                if args.biased_model_type == 'hypothesis_only' or args.biased_model_type == 'both': #hypothesis only
                    self.crap_model2 = AutoModelForSequenceClassification.from_pretrained('./trained_model_hypothesis_only', **task_kwargs)
                    for name, param in self.crap_model2.named_parameters():
                        param.requires_grad = False
                if args.biased_model_type == 'sequence_length':
                    self.crap_model = self.crap_model1
                    self.crap_model1 = None
                if args.biased_model_type == 'hypothesis_only':
                    self.crap_model = self.crap_model2
                    self.crap_model2 = None
                # self.length_model = SentenceLengthModel()

            def forward(self,
                        input_ids = None,
                        attention_mask = None,
                        token_type_ids = None,
                        position_ids = None,
                        head_mask = None,
                        inputs_embeds = None,
                        labels = None,
                        output_attentions = None,
                        output_hidden_states = None,
                        return_dict = None,
                    ):
                # print(ex)
                # print('input ids', input_ids[0])
                # print('shape', input_ids.shape)
                # print('token type ids', token_type_ids[0])
                # print('position ids', position_ids)
                # print('input embeds', inputs_embeds)
                # print(labels)
                # print('removed premise?', remove_premise(input_ids))
                # self.length_model(
                #     input_ids,
                #     attention_mask=attention_mask,
                #     token_type_ids=token_type_ids,
                #     position_ids=position_ids,
                #     head_mask=head_mask,
                #     inputs_embeds=inputs_embeds,
                #     output_attentions=output_attentions,
                #     output_hidden_states=output_hidden_states,
                #     return_dict=return_dict,
                #     labels=labels
                # )

                
                if self.training:
                    self.regular_model.train()
                    if self.crap_model1:
                        self.crap_model1.train()
                    if self.crap_model2:
                        self.crap_model2.train()
                    if self.crap_model:
                        self.crap_model.train()
                    if args.biased_model_type != 'both':
                        if args.biased_model_type == 'sequence_length':
                            biased_model_input_ids = input_ids
                            biased_model_token_type_ids = token_type_ids
                        else: #hypothesis only
                            biased_model_input_ids = remove_premise(input_ids)
                            biased_model_token_type_ids = torch.zeros_like(token_type_ids)
                        reg_output = self.regular_model(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            labels=labels
                        )
                        crap_output = self.crap_model(
                            biased_model_input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=biased_model_token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            labels=labels
                        )
                        loss_fn = torch.nn.CrossEntropyLoss()
                        logits = torch.add(reg_output.logits, crap_output.logits)
                        loss = loss_fn(logits, labels)
                        return SequenceClassifierOutput(
                            loss = loss, 
                            logits = logits
                        )
                    else:
                        biased_model_input_ids = remove_premise(input_ids)
                        biased_model_token_type_ids = torch.zeros_like(token_type_ids)
                        reg_output = self.regular_model(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            labels=labels
                        )
                        crap_output1 = self.crap_model1(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            labels=labels
                        )
                        crap_output2 = self.crap_model2(
                            biased_model_input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=biased_model_token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            labels=labels
                        )
                        loss_fn = torch.nn.CrossEntropyLoss()
                        logits = torch.add(torch.add(reg_output.logits, crap_output1.logits), crap_output2.logits)
                        loss = loss_fn(logits, labels)
                        return SequenceClassifierOutput(
                            loss = loss, 
                            logits = logits
                        )
                else:
                    self.regular_model.eval()
                    # self.crap_model.eval()
                    reg_output = self.regular_model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        labels=labels
                    )
                    return reg_output
        dummy_config = MyConfig()
        model = Hybrid(dummy_config)
        if training_args.do_eval:
            model.load_state_dict(torch.load(args.model + "pytorch_model.bin"))
    else:
        if args.biased_model_type == 'sequence_length':
            model = SentenceLengthModel()  
            if training_args.do_eval:
                model.load_state_dict(torch.load(args.model + "pytorch_model.bin"))
                print('loaded biased sequence length model')
        else:
            # Here we select the right model fine-tuning head
            model_classes = {'qa': AutoModelForQuestionAnswering,
                            'nli': AutoModelForSequenceClassification}
            model_class = model_classes[args.task]
            # Initialize the model and tokenizer from the specified pretrained model/checkpoint
            model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length, hypothesis_only=args.hypothesis_only)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    # dataset = dataset.filter(lambda ex: ex['label'] != -1 and ex['label'] == 2)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        if args.dataset.endswith('.json'): 
            eval_dataset = dataset
        else: 
            eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    # print(model)
    print(train_dataset_featurized)
    print(tokenizer)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
