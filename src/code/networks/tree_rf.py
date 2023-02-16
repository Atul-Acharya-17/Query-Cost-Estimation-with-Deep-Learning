import numpy as np



class TreeRF():

    def __init__(self, tree_pooler, num_features=1708) -> None:
        self.card_rf = []
        self.cost_rf = []
        self.tree_pooler = tree_pooler
        self.n_features = num_features

    def num_features(self):
        return self.n_features

    def add_estimators(self, cost, card):  
        self.cost_rf.append(cost)
        self.card_rf.append(card)

    def predict(self, plan, use_true=False, use_db_pred=False):

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
        
        right_card = np.array([1.0])
        right_cost = np.array([0.0])
        left_card = np.array([1.0])
        left_cost = np.array([0.0])

        has_left = np.array([0])
        has_right = np.array([0])

        if len(plan.children) == 1: #  Only left child
            left_cost, left_card = self.predict(plan.children[0], use_db_pred=use_db_pred, use_true=use_true)
            has_left = np.array([1])

        elif len(plan.children) == 2: # 2 children
            left_cost, left_card = self.predict(plan.children[0], use_db_pred=use_db_pred, use_true=use_true)
            right_cost, right_card = self.predict(plan.children[1], use_db_pred=use_db_pred, use_true=use_true)
            has_left = np.array([1])
            has_right = np.array([1])

        if use_db_pred:
            left_card = np.array([plan.children[0].db_estimate_card]) if len(plan.children) >= 1 else np.array([1])
            right_card = np.array([plan.children[1].db_estimate_card]) if len(plan.children) == 2 else np.array([1])
        
        elif use_true:
            left_card = np.array([plan.children[0].cardinality]) if len(plan.children) >= 1 else np.array([1])
            right_card = np.array([plan.children[1].cardinality]) if len(plan.children) == 2 else np.array([1])

        # data = np.concatenate((op_type, feature, bitmap, condition1_vector, condition2_vector, left_cost, right_cost, left_card, right_card))
        
        data = np.concatenate((op_type, feature, bitmap, condition1_vector, condition2_vector, left_cost, right_cost, left_card, right_card, has_left, has_right))
        data = data.reshape((1, -1))

        cost_sum = 0
        card_sum = 0

        for model in self.cost_rf:
            cost = model.predict(data)
            cost_sum += cost
        cost_pred = cost_sum / len(self.cost_rf)


        if use_db_pred:
            card_pred = plan.db_estimate_card

        elif use_true:
            card_pred = plan.cardinality

        else:
            for model in self.card_rf:
                card = model.predict(data)
                card_sum += card

            card_pred = card_sum / len(self.card_rf)

        return cost_pred, card_pred