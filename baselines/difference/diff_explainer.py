from explainer import Archipelago


class DiffExplainer(Archipelago):
    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=2,
    ):
        Archipelago.__init__(
            self, model, input, baseline, data_xformer, output_indices, batch_size
        )

    def difference_attribution(self, set_indices):
        """
        Gets attributions of index sets by f(x*) - f(x'_{I} + x*_{\I})
        """
        if not set_indices:
            return dict()
        scores = self.batch_set_inference(
            set_indices, self.input, self.baseline, include_context=True
        )
        ditch_scores = scores["scores"]
        input_score = scores["context_score"]
        set_scores = {}
        for index_tuple in ditch_scores:
            set_scores[index_tuple] = input_score - ditch_scores[index_tuple]
        return set_scores
