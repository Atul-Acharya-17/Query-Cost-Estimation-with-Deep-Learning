import torch


class PredicateNode():

    def __init__(self, op_type, operator, left_value=None, right_value=None, children=None):

        self.op_type = op_type
        self.operator = operator
        self.left_value = left_value
        self.right_value = right_value
        self.children = children

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)


class PredicateNodeVector():
    
    def __init__(self, op_type, operator, bool_operator_vector, comp_operator_vector, left_vector, right_vector) -> None:
        self.bool_operator_vector = bool_operator_vector
        self.comp_operator_vector = comp_operator_vector
        self.left_vector = left_vector
        self.right_vector = right_vector

        self.op_type = op_type
        self.operator = operator

        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_torch_tensor(self):
        return torch.unsqueeze(torch.FloatTensor(self.bool_operator_vector + self.left_vector + self.comp_operator_vector + self.right_vector),0)


class PlanNodeVector():

    def __init__(self, operator_vec, extra_info_vec, condition1_root, condition2_root, sample_vec, has_condition, cost, cardinality) -> None:
        self.operator_vec = operator_vec
        self.extra_info_vec = extra_info_vec
        self.condition1_root = condition1_root
        self.condition2_root = condition2_root
        self.sample_vec = sample_vec
        self.has_cond = has_condition
        self.cost = cost
        self.cardinality = cardinality

        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_torch_operation_vector(self):
        return torch.unsqueeze(torch.FloatTensor(self.operator_vec), 0)

    def get_torch_extra_info_vector(self):
        return torch.unsqueeze(torch.FloatTensor(self.extra_info_vec), 0)

    def get_torch_sample_bitmap_vector(self):
        return torch.unsqueeze(torch.FloatTensor(self.sample_vec), 0)
