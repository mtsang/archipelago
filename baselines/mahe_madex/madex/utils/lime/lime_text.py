################################
# Based on the LIME code repo
################################

import numpy as np
import re


class TextDomainMapper:
    """Maps feature ids to words or word-positions"""

    def __init__(self, indexed_string):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [
                (
                    "%s_%s"
                    % (
                        self.indexed_string.word(x[0]),
                        "-".join(map(str, self.indexed_string.string_position(x[0]))),
                    ),
                    x[1],
                )
                for x in exp
            ]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp

    def visualize_instance_html(
        self, exp, label, div_name, exp_object_name, text=True, opacity=True
    ):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return ""
        text = (
            self.indexed_string.raw_string()
            .encode("utf-8", "xmlcharrefreplace")
            .decode("utf-8")
        )
        text = re.sub(r"[<>&]", "|", text)
        exp = [
            (
                self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1],
            )
            for x in exp
        ]
        all_occurrences = list(
            itertools.chain.from_iterable(
                [itertools.product([x[0]], x[1], [x[2]]) for x in exp]
            )
        )
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        ret = """
            %s.show_raw_text(%s, %d, %s, %s, %s);
            """ % (
            exp_object_name,
            json.dumps(all_occurrences),
            label,
            json.dumps(text),
            div_name,
            json.dumps(opacity),
        )
        return ret


class IndexedString(object):
    """String with various indexes."""

    def __init__(self, raw_string, split_expression=r"\W+", bow=True):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            split_expression: string will be split by this.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
        """
        self.raw = raw_string
        self.as_list = re.split(r"(%s)|$" % split_expression, self.raw)
        self.as_np = np.array(self.as_list)
        non_word = re.compile(r"(%s)|$" % split_expression).match
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]]))
        )
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype="bool")
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return "".join(
                [
                    self.as_list[i] if mask[i] else "UNKWORDZ"
                    for i in range(mask.shape[0])
                ]
            )
        return "".join([self.as_list[v] for v in mask.nonzero()[0]])

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(
                itertools.chain.from_iterable([self.positions[z] for z in words])
            )
        else:
            return self.positions[words]
