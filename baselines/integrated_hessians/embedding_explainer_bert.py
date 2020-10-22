"""
A module that sub-classes the original explainer to explain language
models through an embedding layer.
"""
import tensorflow as tf
import numpy as np
from path_explain.explainers.path_explainer_tf import PathExplainerTF
from tqdm import tqdm


class EmbeddingExplainerTF(PathExplainerTF):
    """
    This class is designed to explain models that use an embedding layer,
    e.g. language models. It is very similar to the original path explainer,
    except that it sums over the embedding dimension at convient places
    to reduce dimensionality.
    """

    def __init__(self, model, embedding_axis=2):
        """
        Initialize the TF explainer class. This class
        will handle both the eager and the
        old-school graph-based tensorflow models.

        Args:
            model: A tf.keras.Model instance if executing eagerly.
                   A tuple (input_tensor, output_tensor) otherwise.
            embedding_dimension: The axis corresponding to the embeddings.
                                 Usually this is 2.
        """
        self.model = model
        self.eager_mode = False
        self.embedding_axis = embedding_axis
        try:
            self.eager_mode = tf.executing_eagerly()
        except AttributeError:
            pass

    def _single_attribution(
        self,
        current_input,
        current_baseline,
        current_alphas,
        num_samples,
        batch_size,
        use_expectation,
        output_index,
        attention_mask,
    ):
        """
        A helper function to compute path
        attributions for a single sample.

        Args:
            current_input: A single sample. Assumes that
                           it is of shape (...) where ...
                           represents the input dimensionality
            baseline: A tensor representing the baseline input.
            current_alphas: Which alphas to use when interpolating
            num_samples: The number of samples to draw
            batch_size: Batch size to input to the model
            use_expectation: Whether or not to sample the baseline
            output_index: Whether or not to index into a given class
        """
        current_input = np.expand_dims(current_input, axis=0)
        current_alphas = tf.reshape(
            current_alphas, (num_samples,) + (1,) * (len(current_input.shape) - 1)
        )

        attribution_array = []
        for j in range(0, num_samples, batch_size):
            number_to_draw = min(batch_size, num_samples - j)

            batch_baseline = self._sample_baseline(
                current_baseline, number_to_draw, use_expectation
            )
            batch_alphas = current_alphas[j : min(j + batch_size, num_samples)]

            reps = np.ones(len(current_input.shape)).astype(int)
            reps[0] = number_to_draw
            batch_input = tf.convert_to_tensor(np.tile(current_input, reps))
            batch_attention_mask = np.tile(attention_mask, (number_to_draw, 1))

            batch_attributions = self.accumulation_function(
                batch_input,
                batch_baseline,
                batch_alphas,
                output_index=output_index,
                second_order=False,
                interaction_index=None,
                attention_mask=batch_attention_mask,
            )
            attribution_array.append(batch_attributions)
        attribution_array = np.concatenate(attribution_array, axis=0)
        attributions = np.mean(attribution_array, axis=0)
        return attributions

    def attributions(
        self,
        inputs,
        baseline,
        batch_size=50,
        num_samples=100,
        use_expectation=True,
        output_indices=None,
        verbose=False,
        attention_mask=None,
    ):
        """
        A function to compute path attributions on the given
        inputs.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baseline: A tensor of inputs to the model of shape
                      (num_refs, ...) where ... indicates the dimensionality
                      of the input.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            num_samples: The number of samples to use when computing the
                         expectation or integral.
            use_expectation: If True, this samples baselines and interpolation
                             constants uniformly at random (expected gradients).
                             If False, then this assumes num_refs=1 in which
                             case it uses the same baseline for all inputs,
                             or num_refs=batch_size, in which case it uses
                             baseline[i] for inputs[i] and takes 100 linearly spaced
                             points between baseline and input (integrated gradients).
            output_indices:  If this is None, then this function returns the
                             attributions for each output class. This is rarely
                             what you want for classification tasks. Pass an
                             integer tensor of shape [batch_size] to
                             index the output output_indices[i] for
                             the input inputs[i].
        """
        len(inputs)
        attributions, is_multi_output, num_classes = self._init_array(
            inputs, output_indices, attention_mask
        )

        input_iterable = enumerate(inputs)
        if verbose:
            input_iterable = enumerate(tqdm(inputs))

        for i, current_input in input_iterable:
            current_alphas = self._sample_alphas(num_samples, use_expectation)

            if not use_expectation and baseline.shape[0] > 1:
                current_baseline = np.expand_dims(baseline[i], axis=0)
            else:
                current_baseline = baseline

            if is_multi_output:
                if output_indices is not None:
                    if isinstance(output_indices, int):
                        output_index = output_indices
                    else:
                        output_index = output_indices[i]
                    current_attributions = self._single_attribution(
                        current_input,
                        current_baseline,
                        current_alphas,
                        num_samples,
                        batch_size,
                        use_expectation,
                        output_index,
                        attention_mask[i],
                    )
                    attributions[i] = current_attributions
                else:
                    for output_index in range(num_classes):
                        current_attributions = self._single_attribution(
                            current_input,
                            current_baseline,
                            current_alphas,
                            num_samples,
                            batch_size,
                            use_expectation,
                            output_index,
                            attention_mask[i],
                        )
                        attributions[output_index, i] = current_attributions
            else:
                current_attributions = self._single_attribution(
                    current_input,
                    current_baseline,
                    current_alphas,
                    num_samples,
                    batch_size,
                    use_expectation,
                    None,
                    attention_mask[i],
                )
                attributions[i] = current_attributions
        return attributions

    def accumulation_function(
        self,
        batch_input,
        batch_baseline,
        batch_alphas,
        output_index=None,
        second_order=False,
        interaction_index=None,
        attention_mask=None,
    ):
        """
        A function that computes the logic of combining gradients and
        the difference from reference. This function is meant to
        be overloaded in the case of custom gradient logic. See PathExplainerTF
        for a description of the input.
        """
        if not second_order:
            batch_difference = batch_input - batch_baseline
            batch_interpolated = (
                batch_alphas * batch_input + (1.0 - batch_alphas) * batch_baseline
            )

            with tf.GradientTape() as tape:
                tape.watch(batch_interpolated)

                batch_predictions = self.model(batch_interpolated, attention_mask)
                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]
            batch_gradients = tape.gradient(batch_predictions, batch_interpolated)
            batch_attributions = batch_gradients * batch_difference

            ################################
            # This line is the only difference
            # for attributions. We sum over the embedding dimension.
            batch_attributions = tf.reduce_sum(
                batch_attributions, axis=self.embedding_axis
            )
            ################################

            return batch_attributions

        batch_alpha, batch_beta = batch_alphas
        batch_difference = batch_input - batch_baseline
        batch_interpolated_beta = (
            batch_beta * batch_input + (1.0 - batch_beta) * batch_baseline
        )

        with tf.GradientTape() as second_order_tape:
            second_order_tape.watch(batch_interpolated_beta)

            batch_difference_beta = batch_interpolated_beta - batch_baseline
            batch_interpolated_alpha = (
                batch_alpha * batch_interpolated_beta
                + (1.0 - batch_alpha) * batch_baseline
            )
            with tf.GradientTape() as first_order_tape:
                first_order_tape.watch(batch_interpolated_alpha)

                batch_predictions = self.model(batch_interpolated_alpha, attention_mask)
                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]

            batch_gradients = first_order_tape.gradient(
                batch_predictions, batch_interpolated_alpha
            )
            batch_gradients = batch_gradients * batch_difference_beta

            if interaction_index is not None:
                batch_gradients = tf.reduce_sum(
                    batch_gradients[tuple([slice(None)] + interaction_index)], axis=-1
                )
            #                 print("bg" ,batch_gradients.shape)
            #                 print("BG", batch_gradients)
            else:
                ################################
                # The first of two modifications for interactions.
                batch_gradients = tf.reduce_sum(
                    batch_gradients, axis=self.embedding_axis
                )
                ################################

        #                 print("BG", batch_gradients[:,interaction_index])
        #         assert(False)

        if interaction_index is not None:
            batch_hessian = second_order_tape.gradient(
                batch_gradients, batch_interpolated_beta
            )
        #             print("bh" ,batch_hessian.shape)
        #             print("BH" ,tf.reduce_sum(batch_hessian, axis=2))

        else:
            batch_hessian = second_order_tape.batch_jacobian(
                batch_gradients, batch_interpolated_beta
            )
        #             print("bh" ,batch_hessian.shape)
        #             print("BH" ,tf.reduce_sum(batch_hessian, axis=3)[ :, 1, : ])

        #             print("BH" ,tf.reduce_sum(batch_hessian, axis=2))

        if interaction_index is not None:
            pass
        #             print("bd 0", batch_difference.shape)

        #             batch_difference = batch_difference[tuple([slice(None)] + \
        #                                                                  interaction_index)]
        #             batch_difference = tf.expand_dims(batch_difference, axis=1)

        #             print("bd a", batch_difference.shape)

        #             print("bd1", batch_difference.shape)

        #             print("BD", tf.reduce_sum(batch_difference, axis=2))

        #             for _ in range(len(batch_input.shape) - 1):
        #                 batch_difference = tf.expand_dims(batch_difference, axis=-1)
        #             print("bd b", batch_difference.shape)
        else:
            batch_difference = tf.expand_dims(batch_difference, axis=1)
        #             print("bd1", batch_difference.shape)

        #             print("BD", tf.reduce_sum(batch_difference, axis=3)[ :, :, 1])

        #         print("pre bi", batch_hessian.shape, batch_difference.shape)

        batch_interactions = batch_hessian * batch_difference

        #         if interaction_index is None:
        #             print("BI",tf.reduce_sum(batch_interactions[:,:,1,:], axis=2))
        #             print("BI_2",tf.reduce_sum(batch_interactions[:,1,:,:], axis=2))

        # #             print("bi", batch_interactions.shape)
        #         else:
        #             print("BI", tf.reduce_sum(batch_interactions, axis=2))

        ################################
        # The second of two modifications for interactions.
        if interaction_index is None:
            # This axis computation is really len(input.shape) - 1 + self.embedding_axis - 1
            # The -1's are because we squashed a batch dimension and the first embedding dimension.

            hessian_embedding_axis = len(batch_input.shape) + self.embedding_axis - 2
            batch_interactions = tf.reduce_sum(
                batch_interactions, axis=hessian_embedding_axis
            )
        ################################
        else:
            hessian_embedding_axis = len(batch_input.shape) + self.embedding_axis - 3

            #             print("bi", batch_interactions.shape)
            batch_interactions = tf.reduce_sum(
                batch_interactions, axis=hessian_embedding_axis
            )

        return batch_interactions

    def _single_interaction(
        self,
        current_input,
        current_baseline,
        current_alphas,
        num_samples,
        batch_size,
        use_expectation,
        output_index,
        interaction_index,
        attention_mask,
    ):
        """
        A helper function to compute path
        interactions for a single sample.

        Args:
            current_input: A single sample. Assumes that
                           it is of shape (...) where ...
                           represents the input dimensionality
            baseline: A tensor representing the baseline input.
            current_alphas: Which alphas to use when interpolating
            num_samples: The number of samples to draw
            batch_size: Batch size to input to the model
            use_expectation: Whether or not to sample the baseline
            output_index: Whether or not to index into a given class
            interaction_index: The index to take the interactions with respect to.
        """
        current_input = np.expand_dims(current_input, axis=0)
        current_alpha, current_beta = current_alphas
        current_alpha = tf.reshape(
            current_alpha, (num_samples,) + (1,) * (len(current_input.shape) - 1)
        )
        current_beta = tf.reshape(
            current_beta, (num_samples,) + (1,) * (len(current_input.shape) - 1)
        )
        attribution_array = []
        for j in range(0, num_samples, batch_size):
            number_to_draw = min(batch_size, num_samples - j)

            batch_baseline = self._sample_baseline(
                current_baseline, number_to_draw, use_expectation
            )
            batch_alpha = current_alpha[j : min(j + batch_size, num_samples)]
            batch_beta = current_beta[j : min(j + batch_size, num_samples)]

            reps = np.ones(len(current_input.shape)).astype(int)
            reps[0] = number_to_draw
            batch_input = tf.convert_to_tensor(np.tile(current_input, reps))
            batch_attention_mask = np.tile(attention_mask, (number_to_draw, 1))

            batch_attributions = self.accumulation_function(
                batch_input,
                batch_baseline,
                batch_alphas=(batch_alpha, batch_beta),
                output_index=output_index,
                second_order=True,
                interaction_index=interaction_index,
                attention_mask=batch_attention_mask,
            )
            #             print("BA", batch_attributions.shape)
            #             print("BA mat", batch_attributions)
            attribution_array.append(batch_attributions)
        attribution_array = np.concatenate(attribution_array, axis=0)
        attributions = np.mean(attribution_array, axis=0)
        return attributions

    def interactions(
        self,
        inputs,
        baseline,
        batch_size=50,
        num_samples=100,
        use_expectation=True,
        output_indices=None,
        verbose=False,
        interaction_index=None,
        attention_mask=None,
    ):
        """
        A function to compute path interactions (attributions of
        attributions) on the given inputs.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baseline: A tensor of inputs to the model of shape
                      (num_refs, ...) where ... indicates the dimensionality
                      of the input.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            num_samples: The number of samples to use when computing the
                         expectation or integral.
            use_expectation: If True, this samples baselines and interpolation
                             constants uniformly at random (expected gradients).
                             If False, then this assumes num_refs=1 in which
                             case it uses the same baseline for all inputs,
                             or num_refs=batch_size, in which case it uses
                             baseline[i] for inputs[i] and takes 100 linearly spaced
                             points between baseline and input (integrated gradients).
            output_indices:  If this is None, then this function returns the
                             attributions for each output class. This is rarely
                             what you want for classification tasks. Pass an
                             integer tensor of shape [batch_size] to
                             index the output output_indices[i] for
                             the input inputs[i].
            interaction_index: Either None or an index into the input. If the latter,
                               will compute the interactions with respect to that
                               feature. This parameter should index into a batch
                               of inputs as inputs[(slice(None) + interaction_index)].
                               For example, if you had images of shape (32, 32, 3)
                               and you wanted interactions with respect
                               to pixel (i, j, c), you should pass
                               interaction_index=[i, j, c].
        """
        interactions, is_multi_output, num_classes = self._init_array(
            inputs, output_indices, attention_mask, interaction_index, True
        )

        interaction_index = self._clean_index(interaction_index)

        input_iterable = enumerate(inputs)
        if verbose:
            input_iterable = enumerate(tqdm(inputs))

        for i, current_input in input_iterable:
            current_alphas = self._sample_alphas(
                num_samples, use_expectation, use_product=True
            )

            if not use_expectation and baseline.shape[0] > 1:
                current_baseline = np.expand_dims(baseline[i], axis=0)
            else:
                current_baseline = baseline

            if is_multi_output:
                if output_indices is not None:
                    if isinstance(output_indices, int):
                        output_index = output_indices
                    else:
                        output_index = output_indices[i]
                    current_interactions = self._single_interaction(
                        current_input,
                        current_baseline,
                        current_alphas,
                        num_samples,
                        batch_size,
                        use_expectation,
                        output_index,
                        interaction_index,
                        attention_mask[i],
                    )
                    interactions[i] = current_interactions
                else:
                    for output_index in range(num_classes):
                        current_interactions = self._single_interaction(
                            current_input,
                            current_baseline,
                            current_alphas,
                            num_samples,
                            batch_size,
                            use_expectation,
                            output_index,
                            interaction_index,
                            attention_mask[i],
                        )
                        interactions[output_index, i] = current_interactions
            else:
                current_interactions = self._single_interaction(
                    current_input,
                    current_baseline,
                    current_alphas,
                    num_samples,
                    batch_size,
                    use_expectation,
                    None,
                    interaction_index,
                    attention_mask[i],
                )
                interactions[i] = current_interactions
        return interactions

    def _get_test_output(self, inputs, attention_mask):
        """
        Internal helper function to get the
        output of a model. Designed to
        be overloaded.

        Args:
            inputs: Inputs to the model
        """
        return self.model(inputs[0:1], attention_mask[0:1])

    def _init_array(
        self,
        inputs,
        output_indices,
        attention_mask,
        interaction_index=None,
        as_interactions=False,
    ):
        """
        Internal helper function to get an
        array of the proper shape. This needs
        to be overloaded because the input shape is the
        embedding size, but we will be squashing the embedding dimensions.
        """
        test_output = self._get_test_output(inputs, attention_mask)
        is_multi_output = len(test_output.shape) > 1
        shape_tuple = inputs.shape[:2]
        num_classes = test_output.shape[-1]

        if as_interactions and interaction_index is None:
            shape_tuple = [inputs.shape[0], inputs.shape[1], inputs.shape[1]]
            shape_tuple = tuple(shape_tuple)

        if is_multi_output and output_indices is None:
            num_classes = test_output.shape[-1]
            attributions = np.zeros((num_classes,) + shape_tuple)
        elif not is_multi_output and output_indices is not None:
            raise ValueError(
                "Provided output_indices but " + "model is not multi output!"
            )
        else:
            attributions = np.zeros(shape_tuple)

        return attributions, is_multi_output, num_classes
