import numpy as np


class TreeXGB():

    def __init__(self, tree_pooler) -> None:
        self.card_xgb = []
        self.cost_xgb = []
        self.tree_pooler = tree_pooler

    def add_estimators(self, cost, card):
        self.cost_xgb.append(cost)
        self.card_xgb.append(card)

    def predict(self, plan):

        op_type = np.array(plan.operator_vec)
        feature = np.array(plan.extra_info_vec)
        bitmap = np.array(plan.sample_vec) * plan.has_cond

        cond1 = plan.condition1_root
        cond2 = plan.condition2_root

        if cond1 is None:
            condition1_vector = np.zeros(256)
        else:
            condition1_vector = self.tree_pooler(cond1)[0]
            condition1_vector = condition1_vector.cpu().detach().numpy()

        if cond2 is None:
            condition2_vector = np.zeros(256)
        else:
            condition2_vector = self.tree_pooler(cond2)[0]
            condition2_vector = condition2_vector.cpu().detach().numpy()    
        
        right_card = np.array([1])
        right_cost = np.array([0])
        left_card = np.array([1])
        left_cost = np.array([0])

        if len(plan.children) == 1: #  Only left child
            left_cost, left_card = self.predict(plan.children[0])

        elif len(plan.children) == 2: # 2 children
            left_cost, left_card = self.predict(plan.children[0])
            right_cost, right_card = self.predict(plan.children[1])

        data = np.concatenate((op_type, feature, bitmap, condition1_vector, condition2_vector, left_cost, right_cost, left_card, right_card))
        data = data.reshape((1, -1))

        cost_sum = 0
        card_sum = 0

        for model in self.cost_xgb:
            cost = model.predict(data)
            cost_sum += cost

        for model in self.card_xgb:
            card = model.predict(data)
            card_sum += card

        cost_pred = cost_sum / len(self.cost_xgb)
        card_pred = card_sum / len(self.card_xgb)

        return cost_pred, card_pred