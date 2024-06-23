import os,sys
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from nltk.tokenize import sent_tokenize
import numpy as np



from YoutubeSummarizer.entity.config_entity import DataTransformConfig,ModelTrainingConfig
from YoutubeSummarizer.entity.artifacts_entity import DataTransformArtifact

from YoutubeSummarizer.logger import logging
from YoutubeSummarizer.exception import CustomException

from datasets import load_from_disk

import evaluate

rouge_score = evaluate.load("rouge")

class ModelTrainer:
    def __init__(self,
                 data_transform_artifact=DataTransformArtifact,
                 model_trainer_config=ModelTrainingConfig()):
        self.data_transform_artifact = data_transform_artifact
        self.model_trainer_config = model_trainer_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_trainer_config.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_trainer_config.model_checkpoint)

    def load_data(self,):
        try:
            logging.info("Loading data from disk for tokenization")
            data = load_from_disk(self.data_transform_artifact)
            return data
        except Exception as e:
            raise CustomException(e,sys)
        
    def tokenize_input(self):
        try:

            logging.info("Starting data tokenization, currently insize tokenize_input method of ModelTrainer class")
            tokenizer = self.tokenizer
            def preprocess_function(examples):
                model_inputs = tokenizer(
                    examples["Vid_Transcript"],
                    max_length=self.model_trainer_config.max_input_length,
                    truncation=True,
                )
                labels = tokenizer(
                    examples["Vid_summary"], max_length=self.model_trainer_config.max_target_length, truncation=True
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            data = self.load_data()
            final_data = data.map(preprocess_function, batched =True,
                        remove_columns=['Video_id', 'Vid_Transcript', 'Vid_summary'])
            tokenizer.save()
            return final_data
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def set_arguments(self):
        logging.info("Setting arguments for trainer api of transformers")
        try:
            batch_size = self.model_trainer_config.batch_size
            epochs = self.model_trainer_config.epochs
            model_name = f"{self.model_trainer_config.model_checkpoint}-transcript-summarizer"

            args = Seq2SeqTrainingArguments(
                    model_name,                                                        # Specify Model name
                    eval_strategy="epoch",                                       # Evaluate performance at Epoch end
                    learning_rate=5.6e-5,                            
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    weight_decay=0.01, 
                    save_total_limit=3,                                                # Save the model only 3 times
                    num_train_epochs=epochs,                                           # Number of Epochs
                    predict_with_generate=True,                                        # To predict Sequences
                    fp16=True,                                                         # Set to fp16, mixed precision training
                    push_to_hub=False                                                  # Push model to huggingface
                )
            return args
        except Exception as e:
            raise CustomException(e,sys)
        
    def compute_metrics(self,eval_pred):
        try:
            tokenizer = self.tokenizer
            predictions, labels = eval_pred
            # Decode generated summaries into text
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # Decode reference summaries into text
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # ROUGE expects a newline after each sentence
            decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
            
            # Compute ROUGE scores
            result = rouge_score.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
            # Extract the median scores
            result = {key : value * 100 for key, value in result.items()}
            
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
            
            return {k: round(v,4) for k,v in result.items()}
        
        except Exception as e:
            raise CustomException(e,sys)

    def model_trainer(self):
        logging.info("Starting model training, currently insize model_trainer method of ModelTrainer class")
        try:
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

            trainer = Seq2SeqTrainer(
            self.model,
            self.set_arguments(),
            train_dataset=self.tokenize_input()['train'],
            eval_dataset=self.tokenize_input()['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )

            trainer.train()
            model_trainer_dir = self.model_trainer_config.model_trainer_dir
            self.model.save_pretrained(model_trainer_dir)
            logging.info("Model trained on your dataset")
            logging.info("Exiting model_trainer method of ModelTrainer class")
            return model_trainer_dir
        except Exception as e:
            raise CustomException(e,sys)

