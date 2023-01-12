def q_error(pred, target):
    if pred == 0 and target == 0:
        return 1
    elif pred == 0:
        return target
    elif target == 0:
        return pred
    else:
        return max(pred, target) / min(pred, target)

def mean_squared_error(pred, target):
    return (pred - target)**2

def absolute_error(pred, target):
    return abs(pred - target)
