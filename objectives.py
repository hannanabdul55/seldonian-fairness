
def tpr_rate(A_idx=None, A_val=None):

    def ghat_tpr(X, y_true, y_pred):
        if A_idx is not None and A_val is not None:
            tpr = ((X[:, A_idx] == A_val) & ((y_true == 1) & (y_pred == 1))).astype(int)
        else:
            tpr = ((y_true == 1) & (y_pred == 1)).astype(int)
        return tpr
    return ghat_tpr
