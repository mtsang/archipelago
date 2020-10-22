import tensorflow as tf
from application_utils.text_utils import (
    prepare_huggingface_data,
    get_input_baseline_ids,
)
from transformers import glue_convert_examples_to_features
import numpy as np
from tqdm import tqdm


class BertWrapper:
    def __init__(self, model):
        self.model = model

    def get_predictions(self, batch_ids):
        X = {"input_ids": np.array(batch_ids)}
        batch_conf = self.model(X)[0]
        return batch_conf

    def __call__(self, batch_ids):
        batch_predictions = self.get_predictions(batch_ids)
        return batch_predictions.numpy()


class BertWrapperIH:
    def __init__(self, model):
        self.model = model

    def embedding_model(self, batch_ids):
        batch_embedding = self.model.bert.embeddings((batch_ids, None, None, None))
        #        batch_embedding = self.model.bert.embeddings(batch_ids, None, None, None)
        return batch_embedding

    def prediction_model(self, batch_embedding, attention_mask):
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.model.bert.num_hidden_layers

        encoder_outputs = self.model.bert.encoder(
            [batch_embedding, extended_attention_mask, head_mask], training=False
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)
        logits = self.model.classifier(pooled_output)
        return logits

    def get_predictions(self, batch_ids):
        X = {"input_ids": np.array(batch_ids)}
        batch_conf = self.model(X)[0]
        return batch_conf

    def get_predictions_extra(self, sentences, tokenizer, baseline_token=None):
        X = prepare_huggingface_data(sentences, tokenizer)

        assert len(sentences) == 1
        if baseline_token is not None:
            _, baseline_ids = get_input_baseline_ids(
                sentences[0], baseline_token, tokenizer
            )

        for key in X:
            X[key] = tf.convert_to_tensor(X[key])
        batch_ids = X["input_ids"]
        attention_mask = X["attention_mask"]

        batch_conf = self.model(X)[0]

        batch_embedding = self.embedding_model(batch_ids)
        batch_predictions = self.prediction_model(batch_embedding, attention_mask)

        if baseline_token is None:
            batch_baseline = np.zeros((1, batch_ids.shape[1]), dtype=np.int64)
        else:
            batch_baseline = np.expand_dims(baseline_ids, 0)
        baseline_embedding = self.embedding_model(batch_baseline)

        orig_token_list = []
        for i in range(batch_ids.shape[0]):
            ids = batch_ids[i].numpy()
            tokens = tokenizer.convert_ids_to_tokens(ids)
            orig_token_list.append(tokens)

        return (
            batch_predictions,
            orig_token_list,
            batch_embedding,
            baseline_embedding,
            attention_mask,
        )

    def __call__(self, batch_ids):
        batch_predictions = self.get_predictions(batch_ids)
        return batch_predictions.numpy()


class DistilbertWrapperIH(BertWrapperIH):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def embedding_model(self, batch_ids):
        batch_embedding = self.model.distilbert.embeddings(batch_ids)
        return batch_embedding

    def prediction_model(self, batch_embedding, attention_mask):
        #         attention_mask = tf.ones(batch_embedding.shape[:2])
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        head_mask = [None] * self.model.distilbert.num_hidden_layers

        transformer_output = self.model.distilbert.transformer(
            [batch_embedding, attention_mask, head_mask], training=False
        )[0]
        pooled_output = transformer_output[:, 0]
        pooled_output = self.model.pre_classifier(pooled_output)
        logits = self.model.classifier(pooled_output)
        return logits
