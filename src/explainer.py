import numpy as np


class Explainer:
    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        verbose=False,
    ):

        input, baseline = self.arg_checks(input, baseline, data_xformer)

        self.model = model
        self.input = np.squeeze(input)
        self.baseline = np.squeeze(baseline)
        self.data_xformer = data_xformer
        self.output_indices = output_indices
        self.batch_size = batch_size
        self.verbose = verbose

    def arg_checks(self, input, baseline, data_xformer):
        if (input is None) and (data_xformer is None):
            raise ValueError("Either input or data xformer must be defined")

        if input is not None and baseline is None:
            raise ValueError("If input is defined, the baseline must also defined")

        if data_xformer is not None and input is None:
            input = np.ones(data_xformer.num_features).astype(bool)
            baseline = np.zeros(data_xformer.num_features).astype(bool)
        return input, baseline

    def verbose_iterable(self, iterable):
        if self.verbose:
            from tqdm import tqdm

            return tqdm(iterable)
        else:
            return iterable

    def batch_set_inference(
        self, set_indices, context, insertion_target, include_context=False
    ):
        """
        Creates archipelago type data instances and runs batch inference on them
        All "sets" are represented as tuples to work as keys in dictionaries
        """

        num_batches = int(np.ceil(len(set_indices) / self.batch_size))

        scores = {}
        for b in self.verbose_iterable(range(num_batches)):
            batch_sets = set_indices[b * self.batch_size : (b + 1) * self.batch_size]
            data_batch = []
            for index_tuple in batch_sets:
                new_instance = context.copy()
                for i in index_tuple:
                    new_instance[i] = insertion_target[i]

                if self.data_xformer is not None:
                    new_instance = self.data_xformer(new_instance)

                data_batch.append(new_instance)

            if include_context and b == 0:
                if self.data_xformer is not None:
                    data_batch.append(self.data_xformer(context))
                else:
                    data_batch.append(context)

            preds = self.model(np.array(data_batch))

            for c, index_tuple in enumerate(batch_sets):
                scores[index_tuple] = preds[c, self.output_indices]
            if include_context and b == 0:
                context_score = preds[-1, self.output_indices]

        output = {"scores": scores}
        if include_context and num_batches > 0:
            output["context_score"] = context_score
        return output


class Archipelago(Explainer):
    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        interactive=False,
        verbose=False,
    ):
        Explainer.__init__(
            self,
            model,
            input,
            baseline,
            data_xformer,
            output_indices,
            batch_size,
            verbose,
        )
        self.inter_sets = None
        self.main_effects = None
        self.interactive = interactive
        self.interactive_explanations = None
        self.max_interactive_attribution_magnitude = None

        if self.interactive:
            self.cache_interactive_explanations()

    def archattribute(self, set_indices):
        """
        Gets archipelago attributions of index sets
        """
        if not set_indices:
            return dict()
        scores = self.batch_set_inference(
            set_indices, self.baseline, self.input, include_context=True
        )
        set_scores = scores["scores"]
        baseline_score = scores["context_score"]
        for index_tuple in set_scores:
            set_scores[index_tuple] -= baseline_score
        return set_scores

    def archdetect(
        self,
        get_main_effects=True,
        get_pairwise_effects=True,
        single_context=False,
        weights=[0.5, 0.5],
    ):
        """
        Detects interactions and sorts them
        Optional: gets archipelago main effects and/or pairwise effects from function reuse
        "Effects" are archattribute scores
        """
        search_a = self.search_feature_sets(
            self.baseline,
            self.input,
            get_main_effects=get_main_effects,
            get_pairwise_effects=get_pairwise_effects,
        )
        inter_a = search_a["interactions"]

        # notice that input and baseline have swapped places in the arg list
        search_b = self.search_feature_sets(self.input, self.baseline)
        inter_b = search_b["interactions"]

        inter_strengths = {}
        for pair in inter_a:
            if single_context:
                inter_strengths[pair] = inter_b[pair] ** 2
            else:
                inter_strengths[pair] = (
                    weights[1] * inter_a[pair] ** 2 + weights[0] * inter_b[pair] ** 2
                )
        sorted_scores = sorted(inter_strengths.items(), key=lambda kv: -kv[1])

        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        return output

    def explain(self, top_k=None, separate_effects=False):
        if (self.inter_sets is None) or (self.main_effects is None):
            detection_dict = self.archdetect(get_pairwise_effects=False)
            inter_strengths = detection_dict["interactions"]
            self.main_effects = detection_dict["main_effects"]
            self.inter_sets, _ = zip(*inter_strengths)

        if isinstance(top_k, int):
            thresholded_inter_sets = self.inter_sets[:top_k]
        elif top_k is None:
            thresholded_inter_sets = self.inter_sets
        else:
            raise ValueError("top_k must be int or None")

        inter_sets_merged = merge_overlapping_sets(thresholded_inter_sets)
        inter_effects = self.archattribute(inter_sets_merged)

        if separate_effects:
            return inter_effects, self.main_effects

        merged_indices = merge_overlapping_sets(
            set(self.main_effects.keys()) | set(inter_effects.keys())
        )
        merged_explanation = dict()
        for s in merged_indices:
            if s in inter_effects:
                merged_explanation[s] = inter_effects[s]
            elif s[0] in self.main_effects:
                assert len(s) == 1
                merged_explanation[s] = self.main_effects[s[0]]
            else:
                raise ValueError(
                    "Error: index should have been in either main_effects or inter_effects"
                )
        return merged_explanation

    def search_feature_sets(
        self,
        context,
        insertion_target,
        get_interactions=True,
        get_main_effects=False,
        get_pairwise_effects=False,
    ):
        """
        Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
        "Effects" are archattribute scores
        All three options are combined to reuse function calls
        """
        num_feats = context.size
        idv_indices = [(i,) for i in range(num_feats)]

        preds = self.batch_set_inference(
            idv_indices, context, insertion_target, include_context=True
        )
        idv_scores, context_score = preds["scores"], preds["context_score"]

        output = {}

        if get_interactions:
            pair_indices = []
            pairwise_effects = {}
            for i in range(num_feats):
                for j in range(i + 1, num_feats):
                    pair_indices.append((i, j))

            preds = self.batch_set_inference(pair_indices, context, insertion_target)
            pair_scores = preds["scores"]

            inter_scores = {}
            for i, j in pair_indices:

                # interaction detection
                ell_i = np.abs(context[i].item() - insertion_target[i].item())
                ell_j = np.abs(context[j].item() - insertion_target[j].item())
                inter_scores[(i, j)] = (
                    1
                    / (ell_i * ell_j)
                    * (
                        context_score
                        - idv_scores[(i,)]
                        - idv_scores[(j,)]
                        + pair_scores[(i, j)]
                    )
                )

                if (
                    get_pairwise_effects
                ):  # leverage existing function calls to compute pairwise effects
                    pairwise_effects[(i, j)] = pair_scores[(i, j)] - context_score

            output["interactions"] = inter_scores

            if get_pairwise_effects:
                output["pairwise_effects"] = pairwise_effects

        if get_main_effects:  # leverage existing function calls to compute main effects
            main_effects = {}
            for i in idv_scores:
                main_effects[i[0]] = idv_scores[i] - context_score
            output["main_effects"] = main_effects

        return output

    def get_interactive_explanations(self):
        if self.interactive_explanations is None:
            assert not self.interactive
            self.cache_interactive_explanations()

        return self.interactive_explanations, self.max_interactive_attribution_magnitude

    def cache_interactive_explanations(self):
        detection_dict = self.archdetect(get_pairwise_effects=False)
        inter_strengths = detection_dict["interactions"]
        inter_strengths = [
            (inter, strength) for inter, strength in inter_strengths if strength > 1e-10
        ]
        self.main_effects = detection_dict["main_effects"]
        self.inter_sets, _ = zip(*inter_strengths)

        existing_inter_sets = set()
        inter_sets_slider = []
        existing_inter_sets_slider = set()

        break_out = False
        for top_k in range(len(self.inter_sets)):
            thresholded_inter_sets = self.inter_sets[:top_k]
            inter_sets_merged = merge_overlapping_sets(thresholded_inter_sets)
            inter_sets_key = tuple(sorted(inter_sets_merged))
            if inter_sets_key not in existing_inter_sets_slider:
                inter_sets_slider.append(inter_sets_key)
                existing_inter_sets_slider.add(inter_sets_key)

            for inter_set in inter_sets_merged:
                if inter_set in existing_inter_sets:
                    continue
                else:
                    existing_inter_sets.add(inter_set)
                    if len(inter_set) == len(self.input):
                        break_out = True
                        break
            if break_out:
                break

        inter_effects = self.archattribute(list(existing_inter_sets))
        self.max_interactive_attribution_magnitude = np.max(
            np.abs(list(inter_effects.values()))
        )

        self.interactive_explanations = []
        for inter_sets_key in inter_sets_slider:
            inter_effects_slider = {
                inter_set: inter_effects[inter_set] for inter_set in inter_sets_key
            }
            merged_indices = merge_overlapping_sets(
                set(self.main_effects.keys()) | set(inter_effects_slider.keys())
            )

            merged_explanation_slider = dict()
            for s in merged_indices:
                if s in inter_effects:
                    merged_explanation_slider[s] = inter_effects[s]
                elif s[0] in self.main_effects:
                    assert len(s) == 1
                    merged_explanation_slider[s] = self.main_effects[s[0]]
                else:
                    raise ValueError(
                        "Error: index should have been in either main_effects or inter_effects"
                    )
            self.interactive_explanations.append(merged_explanation_slider)


def merge_overlapping_sets(lsts, output_ints=False):
    """Check each number in our arrays only once, merging when we find
    a number we have seen before.

    O(N) mergelists5 solution from https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    """

    def locatebin(bins, n):
        """
        Find the bin where list n has ended up: Follow bin references until
        we find a bin that has not moved.
        """
        while bins[n] != n:
            n = bins[n]
        return n

    data = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        data.append(set(lst))

    bins = list(range(len(data)))  # Initialize each bin[n] == n
    nums = dict()

    sets = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        sets.append(set(lst))

    for r, row in enumerate(data):
        for num in row:
            if num not in nums:
                # New number: tag it with a pointer to this row's bin
                nums[num] = r
                continue
            else:
                dest = locatebin(bins, nums[num])
                if dest == r:
                    continue  # already in the same bin

                if dest > r:
                    dest, r = r, dest  # always merge into the smallest bin

                data[dest].update(data[r])
                data[r] = None
                # Update our indices to reflect the move
                bins[r] = dest
                r = dest

    # take single values out of sets
    output = []
    for s in data:
        if s:
            if output_ints and len(s) == 1:
                output.append(next(iter(s)))
            else:
                output.append(tuple(sorted(s)))

    return output
