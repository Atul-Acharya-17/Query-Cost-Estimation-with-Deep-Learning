class Node(object):
    def __init__(self, cardinality=None, cost=None):
        self.cardinality = cardinality
        self.cost = cost

class Materialize(Node):
    def __init__(self, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = 'Materialize'

    def __str__(self):
        return 'Materialize'


class Aggregate(Node):
    def __init__(self, strategy, keys, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = 'Aggregate'
        self.strategy = strategy
        self.group_keys = keys

    def __str__(self):
        return 'Aggregate ON: ' + ','.join(self.group_keys)


class Sort(Node):
    def __init__(self, sort_keys, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.sort_keys = sort_keys
        self.node_type = 'Sort'

    def __str__(self):
        return 'Sort by: ' + ','.join(self.sort_keys)


class Hash(Node):
    def __init__(self, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = 'Hash'

    def __str__(self):
        return 'Hash'


class Join(Node):
    def __init__(self, node_type, condition_seq, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = node_type
        self.condition = condition_seq

    def __str__(self):
        return self.node_type + ' ON ' + ','.join([str(i) for i in self.condition])


class Scan(Node):
    def __init__(self, node_type, condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = node_type
        self.condition_filter = condition_seq_filter
        self.condition_index = condition_seq_index
        self.relation_name = relation_name
        self.index_name = index_name

    def __str__(self):
        return self.node_type + ' ON ' + ','.join([str(i) for i in self.condition_filter]) + '; ' + ','.join(
            [str(i) for i in self.condition_index])


class BitmapCombine(Node):
    def __init__(self, operator, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = operator

    def __str__(self):
        return self.node_type


class Result(Node):
    def __init__(self, cardinality=None, cost=None):
        Node.__init__(self, cardinality=cardinality, cost=cost)
        self.node_type = 'Result'

    def __str__(self):
        return 'Result'