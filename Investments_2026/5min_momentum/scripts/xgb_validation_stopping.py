"""Native XGBoost updates until validation early stopping, without a round cap."""
import math


def train_until_validation_stops(parameters, dtrain, dvalid, patience=30, guard=None,
                                *, metric="mlogloss", maximize=False, score_function=None):
    """Match xgb.train's strict loss-improvement stopping rule, with no max round.

    Uses the same native Booster, training/validation cache, update and evaluation
    operations as xgb.train. There is no finite boosting-round budget. Resource
    interruption, when requested by the caller, leaves earlier checkpoints intact.
    """
    import xgboost as xgb
    if patience < 1:
        raise ValueError("Positive validation patience required.")
    model=xgb.Booster(parameters,cache=[dtrain,dvalid])
    best_loss=-math.inf if maximize else math.inf
    best_iteration=-1
    epoch=0
    history=[]
    while True:
        if guard is not None:
            guard(epoch)
        model.update(dtrain,iteration=epoch)
        if score_function is None:
            result=model.eval_set([(dvalid,"random20")],iteration=epoch)
            entries=dict(item.rsplit(":",1) for item in result.split()[1:])
            score=float(entries["random20-"+metric])
        else:
            score=float(score_function(model.predict(dvalid)))
        if not math.isfinite(score):
            raise ValueError("Nonfinite early-stopping validation loss.")
        history.append(score)
        improved=score>best_loss if maximize else score<best_loss
        if improved:
            best_loss=score
            best_iteration=epoch
            model.set_attr(best_iteration=str(epoch),best_score=str(score))
        if epoch-best_iteration >= patience:
            break
        epoch+=1
    return model,{"random20":{metric:history}}
