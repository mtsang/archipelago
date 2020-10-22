from nltk import ParentedTree


def parse_tree(s):
    tree = ParentedTree.fromstring(s)
    return tree


def read_trees_from_corpus(path):
    f = open(path)
    rows = f.readlines()
    trees = []
    for row in rows:
        row = row.lower()
        tree = parse_tree(row)
        trees.append(tree)
    return trees


def is_leaf(node):
    if type(node[0]) == str and len(node) != 1:
        print(1)
    return type(node[0]) == str


def get_span_to_node_mapping(tree):
    def dfs(node, span_to_node, node_to_span, idx):
        if is_leaf(node):
            span_to_node[idx] = node
            node_to_span[id(node)] = idx
            return idx + 1
        prev_idx = idx
        for child in node:
            idx = dfs(child, span_to_node, node_to_span, idx)
        span_to_node[(prev_idx, idx - 1)] = node
        node_to_span[id(node)] = (prev_idx, idx - 1)
        return idx

    span2node, node2span = {}, {}
    dfs(tree, span2node, node2span, 0)
    return span2node, node2span


def get_siblings_idx(node, node2span):
    parent = node.parent()
    if parent is None:  # root
        return node2span[id(node)]
    return node2span[id(parent)]


def find_region_neighbourhood(s_or_tree, region):
    if type(s_or_tree) is str:
        tree = parse_tree(s_or_tree)
    else:
        tree = s_or_tree

    if type(region) is tuple and region[0] == region[1]:
        region = region[0]

    span2node, node2span = get_span_to_node_mapping(tree)
    node = span2node[region]
    sibling_idx = get_siblings_idx(node, node2span)
    return sibling_idx
