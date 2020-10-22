from explainer import Archipelago
import copy


class ContextExplainer(Archipelago):
    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        verbose=True,
    ):
        super().__init__(
            model, input, baseline, data_xformer, output_indices, batch_size, verbose
        )

    def detect_with_running_contexts(self, contexts):
        """
        Detects interactions and sorts them
        Optional: gets archipelago main effects and/or pairwise effects from function reuse
        """
        inter_scores_each_context = []
        inter_scores_running = {}
        for n, context in enumerate(contexts):
            insertion_target = []
            for i, c in enumerate(context):
                assert c in {self.baseline[i], self.input[i]}
                if c == self.baseline[i]:
                    insertion_target.append(self.input[i])
                else:
                    insertion_target.append(self.baseline[i])

            search = self.search_feature_sets(context, insertion_target)

            context_inters = search["interactions"]

            for pair in context_inters:
                if pair not in inter_scores_running:
                    inter_scores_running[pair] = 0
                inter_scores_running[pair] += context_inters[pair] ** 2

            inter_scores = copy.deepcopy(inter_scores_running)
            for key in inter_scores:
                inter_scores[key] = inter_scores[key] / (n + 1)
            sorted_scores = sorted(inter_scores.items(), key=lambda kv: -kv[1])

            inter_scores_each_context.append(sorted_scores)

        return inter_scores_each_context

    def detect_with_contexts(self, contexts):
        """
        Detects interactions and sorts them
        Optional: gets archipelago main effects and/or pairwise effects from function reuse
        """
        inters = []
        for context in contexts:
            insertion_target = []
            for i, c in enumerate(context):
                assert c in {self.baseline[i], self.input[i]}
                if c == self.baseline[i]:
                    insertion_target.append(self.input[i])
                else:
                    insertion_target.append(self.baseline[i])

            search = self.search_feature_sets(context, insertion_target)
            inters.append(search["interactions"])

        inter_scores = {}
        for pair in inters[0]:
            avg_score = 0
            for inter in inters:
                avg_score += inter[pair] ** 2
            inter_scores[pair] = avg_score / len(inters)
        sorted_scores = sorted(inter_scores.items(), key=lambda kv: -kv[1])

        output = {"interactions": sorted_scores}

        return output
