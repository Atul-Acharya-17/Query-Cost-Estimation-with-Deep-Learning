import numpy as np

import xgboost
import lightgbm
import daal4py as d4p


class TreeGBM():

    def __init__(self, tree_pooler, fast_inference=False, num_features=1706) -> None:
        self.card_gbm = []
        self.cost_gbm = []
        self.tree_pooler = tree_pooler
        self.fast_inference = fast_inference
        self.n_features = num_features

    def num_features(self):
        return self.n_features

    def add_estimators(self, cost, card):
        if self.fast_inference:
            if type(cost) == xgboost.XGBRegressor:
                booster = cost.get_booster()
                setattr(booster, 'num_features', self.num_features)
                cost = d4p.get_gbt_model_from_xgboost(booster)
            elif type(cost) == lightgbm.LGBMRegressor:
                cost = d4p.get_gbt_model_from_lightgbm(cost.booster_)

            if type(card) == xgboost.XGBRegressor:
                booster = card.get_booster()
                setattr(booster, 'num_features', self.num_features)
                card = d4p.get_gbt_model_from_xgboost(booster)
            elif type(card) == lightgbm.LGBMRegressor:
                card = d4p.get_gbt_model_from_lightgbm(card.booster_)            

        self.cost_gbm.append(cost)
        self.card_gbm.append(card)

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
        
        right_card = np.array([1])
        right_cost = np.array([0])
        left_card = np.array([1])
        left_cost = np.array([0])

        if len(plan.children) == 1: #  Only left child
            left_cost, left_card = self.predict(plan.children[0])

        elif len(plan.children) == 2: # 2 children
            left_cost, left_card = self.predict(plan.children[0])
            right_cost, right_card = self.predict(plan.children[1])

        if use_db_pred:
            left_card = np.array([plan.children[0].db_estimate_card]) if len(plan.children) >= 1 else np.array([1])
            right_card = np.array([plan.children[1].db_estimate_card]) if len(plan.children) == 2 else np.array([1])
        
        elif use_true:
            left_card = np.array([plan.children[0].cardinality]) if len(plan.children) >= 1 else np.array([1])
            right_card = np.array([plan.children[1].cardinality]) if len(plan.children) == 2 else np.array([1])
        
        data = np.concatenate((op_type, feature, bitmap, condition1_vector, condition2_vector, left_cost, right_cost, left_card, right_card))
        data = data.reshape((1, -1))

        cost_sum = 0
        card_sum = 0

        for model in self.cost_gbm:
            if self.fast_inference:
                cost = d4p.gbt_regression_prediction().compute(data, model).prediction[0]
            else:
                cost = model.predict(data)
            cost_sum += cost

        for model in self.card_gbm:
            if self.fast_inference:
                card = d4p.gbt_regression_prediction().compute(data, model).prediction[0]
            else:
                card = model.predict(data)

            card_sum += card

        cost_pred = cost_sum / len(self.cost_xgb)
        card_pred = card_sum / len(self.card_xgb)

        return cost_pred, card_pred