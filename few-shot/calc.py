import pandas as pd
import re

EXTRACTED_COLUMNS = {
    'nanozymes': [
        'formula', 'activity', 'syngony', 'length', 'width', 'depth', 'surface',
        'km_value', 'km_unit', 'vmax_value', 'vmax_unit', 'reaction_type',
        'c_min', 'c_max', 'c_const', 'c_const_unit', 'ccat_value', 'ccat_unit',
        'ph', 'temperature'
    ]
}

NUMERIC_COLUMNS = {
    'nanozymes': [
        'length', 'width', 'depth', 'km_value', 'vmax_value', 'c_min', 'c_max',
        'c_const', 'ccat_value', 'ph', 'temperature'
    ]
}

def convert_comma(x):
    try:
        return str(x).replace(',', '.')
    except:
        return str(x)
    
def normalize_colname(col):
    return col.replace("μ", "μ").replace("µ", "μ").strip().lower()

def empty_metrics(cols):
    metrics = dict()
    for col in cols:
        metrics[col] = {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0}
    return pd.DataFrame(metrics).T

def calc_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    metrics = {}
    from copy import deepcopy
    for col in df_true.columns:
        true_values = list(df_true[col].astype(str).values)
        pred_values = list(df_pred[col].astype(str).values)

        tv = deepcopy(true_values)
        pv = deepcopy(pred_values)

        tp = 0
        for val in tv:
            if val in pv:
                pv.pop(pv.index(val))
                tp += 1

        fp = 0
        tv = deepcopy(true_values)
        pv = deepcopy(pred_values)
        for val in pv:
            if val in tv:
                tv.pop(tv.index(val))
            else:
                fp += 1

        fn = 0
        tv = deepcopy(true_values)
        pv = deepcopy(pred_values)
        for val in tv:
            if val in pv:
                pv.pop(pv.index(val))
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[col] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return pd.DataFrame(metrics).T

def load_and_prepare(filename, cols, numeric_cols):
    df = pd.read_csv(filename)
    df.columns = [normalize_colname(str(c)) for c in df.columns]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: convert_comma(x))
    df = df.fillna('NOT_DETECTED')
    return df

def normalize_pdf_key(x):
    x = str(x).strip().lower()
    if x.endswith('.pdf'):
        x = x[:-4]
    return x

def norm_unit(val):
    if pd.isna(val): return 'ND'
    val = str(val).replace('µ', 'μ').replace('/', ' ').replace('-', ' ')
    val = re.sub(r'\s+', ' ', val).strip().lower()
    if 'mol' in val and ('/s' in val or 's-1' in val):
        return 'mol/s'
    if val in ['nd', 'not_detected', 'not detected', 'nan', '']:
        return 'ND'
    return val

def norm_number(val):
    if pd.isna(val): return 'ND'
    val = str(val).strip().lower()
    if val in ['nd', 'not_detected', 'not detected', '', 'nan']:
        return 'ND'
    try:
        v = float(val)
        return round(v, 2)
    except Exception:
        return val

def norm_text(val):
    if pd.isna(val): return 'ND'
    val = str(val).strip().lower().replace('µ', 'μ')
    if val in ['nd', 'not_detected', 'not detected', '', 'nan']:
        return 'ND'
    return val

def compare_predictions_norm(real_file, pred_file, cols, numeric_cols, name=''):
    df_real = load_and_prepare(real_file, cols, numeric_cols)
    df_pred = load_and_prepare(pred_file, cols, numeric_cols)
    # Normalize pdf keys for mapping
    for col in ['pdf']:
        if col in df_real.columns:
            df_real[col] = df_real[col].apply(normalize_pdf_key)
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].apply(normalize_pdf_key)
    # Нормализация данных
    unit_cols = [c for c in cols if 'unit' in c]
    num_cols = list(set(numeric_cols + ['temperature', 'ph', 'length', 'width', 'depth', 'km_value', 'vmax_value', 'c_min', 'c_max', 'c_const', 'ccat_value']))
    for c in cols:
        if c in unit_cols:
            df_real[c] = df_real[c].apply(norm_unit)
            df_pred[c] = df_pred[c].apply(norm_unit)
        elif c in num_cols:
            df_real[c] = df_real[c].apply(norm_number)
            df_pred[c] = df_pred[c].apply(norm_number)
        else:
            df_real[c] = df_real[c].apply(norm_text)
            df_pred[c] = df_pred[c].apply(norm_text)
    key = 'pdf'
    access_articles = list(set(df_real[key].unique()) & set(df_pred[key].unique()))
    metrics_total = empty_metrics(cols)
    for article in access_articles:
        d1 = df_real.loc[df_real[key] == article][cols]
        d2 = df_pred.loc[df_pred[key] == article][cols]
        df_metrics = calc_metrics(d1, d2)
        metrics_total += df_metrics
    metrics_total = metrics_total / len(access_articles) if access_articles else metrics_total
    print(f"\n{name} - обработано статей: {len(access_articles)}")
    return metrics_total

if __name__ == "__main__":
    cols = [normalize_colname(c) for c in EXTRACTED_COLUMNS['nanozymes']]
    numeric_cols = [normalize_colname(c) for c in NUMERIC_COLUMNS['nanozymes']]

    # Zero-shot
    metrics_zero = compare_predictions_norm(
        'real_estate_data.csv', 
        'zero_shot.csv', 
        cols, numeric_cols, name='Zero-shot'
    )
    print("\nZero-shot metrics:\n", metrics_zero)
    metrics_zero.to_csv('metrics_zero_shot.csv')

    # Few-shot
    metrics_few = compare_predictions_norm(
        'real_estate_data.csv', 
        'few_shot.csv', 
        cols, numeric_cols, name='Few-shot'
    )
    print("\nFew-shot metrics:\n", metrics_few)
    metrics_few.to_csv('metrics_few_shot.csv')
