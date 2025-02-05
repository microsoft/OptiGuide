
import numpy as np
import gap_data as _data


########### Helper functions for context extraction ############
def computeRowFeatures1(rows, model):  
    features = []
    for row in rows:
        ft = model.getRowFeatures1(row)
        features.append(ft)
    return features


def computeInputScores(cuts, model):
    score_funcs = [
        model.getCutViolation,
        model.getCutRelViolation,
        model.getCutObjParallelism,
        model.getCutEfficacy,
        model.getCutSCIPScore,
        model.getCutExpImprov,
        model.getCutSupportScore,
        model.getCutIntSupport,
    ]

    scores = np.empty((len(cuts), len(score_funcs)), dtype=np.float32)
    for i, cut in enumerate(cuts):
        for j, score_func in enumerate(score_funcs):
            scores[i, j] = score_func(cut)

    return scores


def computeColFeatures1(cols, model):  
    features = []
    for col in cols:
        ft = model.getColFeatures1(col)
        features.append(ft)
    return features


def computeCoefs(rows, cols, model):
    # hash col position for fast retrieval..
    col_dict = {}
    for j, col in enumerate(cols):
        colname = col.getVar().name
        assert not (colname in col_dict)
        col_dict[colname] = j

    coefs = {}
    for (i, row) in enumerate(rows):
        row_cols = row.getCols()
        row_js = [col_dict[col.getVar().name] for col in row_cols if col.getVar().name in col_dict]
        row_coefs = [val for val, col in zip(row.getVals(), row_cols) if col.getVar().name in col_dict]
        coefs[i] = (row_js, row_coefs)

    return coefs
################################################################



def getContext(model):
    rows = model.getLPRowsData()
    cols = model.getLPColsData()

    # constraints
    row_features = computeRowFeatures1(rows, model)
    row_input_scores = computeInputScores(rows, model)
    row_coefs = computeCoefs(rows, cols, model)

    # variables
    col_features = computeColFeatures1(cols, model)

    raw_data = {
        'row_features.pkl': row_features,
        'row_input_scores.npy': row_input_scores, 
        'col_features.pkl': col_features,
        'row_coefs.pkl': row_coefs,
    }

    processed_data = _data.MyData.from_rawdata(raw_data)
    inp = _data.MyData(*processed_data)
    
    return inp
