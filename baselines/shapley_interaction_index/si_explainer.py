import numpy as np
from tqdm import tqdm
from itertools import chain, combinations
import math, random
from explainer import Explainer


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def random_subset(s):
    out = []
    for el in s:
        # random coin flip
        if random.randint(0, 1) == 0:
            out.append(el)
    return tuple(out)


class SiExplainer(Explainer):
    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        verbose=False,
        seed=None,
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
        if seed is not None:
            random.seed(seed)

    def attribution(self, S, num_T):
        """
        S: the interaction index set to get attributions for
        T: the input index set
        """

        s = len(S)
        n = len(self.input)

        N_excl_S = [i for i in range(n) if i not in S]

        num_T = min(num_T, 2 ** len(N_excl_S))

        random_T_set = set()
        for _ in range(num_T):
            T = random_subset(N_excl_S)
            while T in random_T_set:
                T = random_subset(N_excl_S)
            random_T_set.add(T)

        total_att = 0

        for T in random_T_set:
            t = len(T)

            n1 = math.factorial(n - t - s)
            n2 = math.factorial(t)
            d1 = math.factorial(n - s + 1)

            coef = (n1 * n2) / d1

            subsetsW = powerset(S)

            set_indices = []
            for W in subsetsW:
                set_indices.append(tuple(set(W) | set(T)))

            scores_dict = self.batch_set_inference(
                set_indices, self.baseline, self.input, include_context=False
            )
            scores = scores_dict["scores"]

            att = 0
            for i, W in enumerate(subsetsW):
                w = len(W)
                att += (-1) ** (w - s) * scores[set_indices[i]]

            total_att += coef * att

        return total_att

    def batch_attribution(self, num_T, main_effects=False, pairwise=True):
        """
        S: the interaction index set to get attributions for
        T: the input index set
        """

        def collect_att(S, S_T_Z_dict, Z_score_dict, n):
            s = len(S)

            subsetsW = powerset(S)

            total_att = 0

            for T in S_T_Z_dict[S]:

                att = 0
                for i, W in enumerate(subsetsW):
                    w = len(W)
                    att += (-1) ** (w - s) * Z_score_dict[S_T_Z_dict[S][T][i]]

                t = len(T)
                n1 = math.factorial(n - t - s)
                n2 = math.factorial(t)
                d1 = math.factorial(n - s + 1)

                coef = (n1 * n2) / d1
                total_att += coef * att

            return total_att

        n = len(self.input)
        num_features = n

        if main_effects == False and pairwise == False:
            raise ValueError()
        if main_effects == True and pairwise == True:
            raise ValueError()

        Ss = []
        if pairwise:
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    S = (i, j)
                    Ss.append(S)
        elif main_effects:
            for i in range(num_features):
                Ss.append(tuple([i]))

        Z_set = set()
        S_T_Z_dict = {}

        for S in Ss:
            s = len(S)

            N_excl_S = [i for i in range(n) if i not in S]
            num_T = min(num_T, 2 ** len(N_excl_S))

            random_T_set = set()
            for _ in range(num_T):
                T = random_subset(N_excl_S)
                while T in random_T_set:
                    T = random_subset(N_excl_S)
                random_T_set.add(tuple(T))

            S_T_Z_dict[S] = {}

            subsetsW = powerset(S)

            for T in random_T_set:
                S_T_Z_dict[S][T] = []

                for W in subsetsW:
                    Z = tuple(set(W) | set(T))
                    Z_set.add(Z)
                    S_T_Z_dict[S][T].append(Z)

        Z_list = list(Z_set)
        scores_dict = self.batch_set_inference(
            Z_list, self.baseline, self.input, include_context=False
        )
        scores = scores_dict["scores"]
        Z_score_dict = scores

        if pairwise:
            res = np.zeros((num_features, num_features))
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    S = (i, j)
                    att = collect_att(S, S_T_Z_dict, Z_score_dict, n)
                    res[i, j] = att
            return res

        elif main_effects:
            res = []
            for i in range(num_features):
                S = tuple([i])
                att = collect_att(S, S_T_Z_dict, Z_score_dict, n)
                res.append(att)

            return np.array(res)
