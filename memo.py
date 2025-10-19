# preprocess_and_experiments.py
# Unifies: augmentation (RF only), HOG, landmarks, HOG+landmarks, class weights (MLP),
# SMOTE (RF/SVM), standard pipeline (scaler + low-variance + high-corr)

import os, math, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

# --- Feature libs ---
import cv2
from skimage.feature import hog
import dlib  # need shape_predictor_68_face_landmarks.dat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -----------------------
# 0) Data already split by you -> functions expect:
#    X_* : np.ndarray of shape (N, 48, 48) floats in [0,1]
#    y_* : np.ndarray of labels
# -----------------------

# -----------------------
# 1) Augmentation (image-level; apply to RF training only if enabled)
# -----------------------
def augment_images(images: np.ndarray, y: np.ndarray, times:int=1,
                   rot_deg:float=10, shift_px:int=3, hflip:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    H=W=48
    aug_X, aug_y = [images], [y]
    for _ in range(times):
        batch = []
        for img in images:
            m = img.copy()
            # random rotation
            angle = rng.uniform(-rot_deg, rot_deg)
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
            m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # random shift
            tx, ty = rng.integers(-shift_px, shift_px+1, size=2)
            M = np.float32([[1,0,tx],[0,1,ty]])
            m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # random horizontal flip
            if hflip and rng.random() < 0.5:
                m = np.fliplr(m)
            batch.append(m)
        aug_X.append(np.stack(batch))
        aug_y.append(y.copy())
    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)

# -----------------------
# 2) HOG transformer
# -----------------------
class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys"):
        self.kw = dict(orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm=block_norm, transform_sqrt=True, feature_vector=True)
    def fit(self, X, y=None): return self
    def transform(self, X):
        feats = [hog(x.astype(np.float32), **self.kw) for x in X]
        return np.asarray(feats)

# -----------------------
# 3) Landmark transformer (dlib 68 points -> normalized coordinates)
# -----------------------
class LandmarkTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, shape_predictor_path:str, detect_factor:float=1.0):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = shape_predictor_path
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detect_factor = detect_factor
    def fit(self, X, y=None): return self
    def _one(self, img: np.ndarray) -> np.ndarray:
        img_u8 = (img*255).astype(np.uint8) if img.max()<=1.0 else img.astype(np.uint8)
        rects = self.detector(img_u8, 1)
        if len(rects)==0:
            # fallback: full frame box
            rects = [dlib.rectangle(0,0,img.shape[1]-1,img.shape[0]-1)]
        rect = rects[0]
        shape = self.predictor(img_u8, rect)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
        # normalize to [0,1] by bbox
        x0,y0,x1,y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        w = max(1, x1-x0); h = max(1, y1-y0)
        pts_norm = (pts - np.array([x0,y0])) / np.array([w,h])
        return pts_norm.reshape(-1)
    def transform(self, X):
        feats = [self._one(x) for x in X]
        return np.asarray(feats)

# -----------------------
# 4) Combine features: simple concat of selected transformers
# -----------------------
class FeatureConcat(BaseEstimator, TransformerMixin):
    def __init__(self, use_hog=True, use_lmk=True, hog_params=None, lmk_predictor_path=None):
        self.use_hog = use_hog
        self.use_lmk = use_lmk
        self.hog_params = hog_params or {}
        self.lmk_predictor_path = lmk_predictor_path
    def fit(self, X, y=None):
        self.hog_t = HOGTransformer(**self.hog_params) if self.use_hog else None
        self.lmk_t = LandmarkTransformer(self.lmk_predictor_path) if self.use_lmk else None
        if self.hog_t: self.hog_t.fit(X, y)
        if self.lmk_t: self.lmk_t.fit(X, y)
        return self
    def transform(self, X):
        parts = []
        if self.hog_t: parts.append(self.hog_t.transform(X))
        if self.lmk_t: parts.append(self.lmk_t.transform(X))
        if not parts: raise ValueError("No feature enabled")
        return np.concatenate(parts, axis=1) if len(parts)>1 else parts[0]

# -----------------------
# 5) Correlation filter (drop highly correlated features)
# -----------------------
class CorrFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold:float=0.95):
        self.threshold = threshold
        self.keep_idx_: Optional[np.ndarray] = None
    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        corr = Xdf.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.keep_idx_ = np.array([i for i in range(Xdf.shape[1]) if i not in set(drop)], dtype=int)
        return self
    def transform(self, X):
        return X[:, self.keep_idx_] if self.keep_idx_ is not None else X

# -----------------------
# 6) Class weights for MLP via sample_weight
# -----------------------
def make_sample_weights(y: np.ndarray, classes: List[int]) -> np.ndarray:
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    wmap = {c: w for c, w in zip(classes, cw)}
    return np.array([wmap[int(t)] for t in y], dtype=np.float32)

# -----------------------
# 7) Standard pipeline: scaler + low variance + high correlation
# -----------------------
def make_standard_preproc(var_thresh:float=0.0, corr_thresh:float=0.95):
    steps = []
    steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    if var_thresh > 0.0:
        steps.append(("var", VarianceThreshold(var_thresh)))
    steps.append(("corr", CorrFilter(threshold=corr_thresh)))
    return steps

# -----------------------
# 8) Build model pipelines per recipe
# -----------------------
@dataclass
class Config:
    use_hog: bool=True
    use_lmk: bool=False
    aug_for_rf: bool=False
    smote: bool=False           # only for RF/SVM
    model: str="svm"            # "svm" | "rf" | "mlp"
    var_thresh: float=0.0
    corr_thresh: float=0.95
    hog_params: Dict[str,Any]=None
    shape_predictor_path: Optional[str]=None
    mlp_params: Dict[str,Any]=None
    rf_params: Dict[str,Any]=None
    svm_params: Dict[str,Any]=None

def build_pipeline(cfg: Config):
    feature = FeatureConcat(use_hog=cfg.use_hog,
                            use_lmk=cfg.use_lmk,
                            hog_params=cfg.hog_params or {},
                            lmk_predictor_path=cfg.shape_predictor_path)

    preproc = make_standard_preproc(cfg.var_thresh, cfg.corr_thresh)

    if cfg.model == "rf":
        clf = RandomForestClassifier(**(cfg.rf_params or dict(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)))
        # RF + optional SMOTE
        if cfg.smote:
            pipe = ImbPipeline([("feat", feature)] + preproc + [("smote", SMOTE(random_state=42)), ("clf", clf)])
        else:
            pipe = ImbPipeline([("feat", feature)] + preproc + [("clf", clf)])
        return pipe

    if cfg.model == "svm":
        clf = SVC(probability=False, **(cfg.svm_params or dict(C=2.0, gamma=1e-2, kernel="rbf", class_weight=None, random_state=42)))
        pipe = ImbPipeline([("feat", feature)] + preproc + ([("smote", SMOTE(random_state=42))] if cfg.smote else []) + [("clf", clf)])
        return pipe

    if cfg.model == "mlp":
        # MLP has no class_weight; pass sample_weight during fit
        clf = MLPClassifier(**(cfg.mlp_params or dict(hidden_layer_sizes=(256,128), alpha=1e-4, learning_rate_init=1e-3,
                                                      batch_size=64, max_iter=100, random_state=42)))
        pipe = ImbPipeline([("feat", feature)] + preproc + [("clf", clf)])
        return pipe

    raise ValueError("Unknown model")

# -----------------------
# 9) Train/eval helper
# -----------------------
def train_and_eval(pipe, X_tr, y_tr, X_va, y_va, model_name:str, use_mlp_weights:bool=False, classes=None):
    if use_mlp_weights:
        sw = make_sample_weights(y_tr, classes=classes if classes is not None else np.unique(y_tr))
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
    else:
        pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    f1 = f1_score(y_va, y_pred, average="macro")
    print(f"{model_name:4s} | F1={f1:.4f}")
    print(classification_report(y_va, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_va, y_pred))
    return pipe, f1

# -----------------------
# 10) Recipes mapping to你的 1–7
# -----------------------
def run_recipes(X_train, y_train, X_val, y_val, shape_predictor_path:str):
    results = {}

    # Common HOG params used in your pipeline
    hog_params = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys")

    # 1) augmentation (RF only)
    Xtr_rf, ytr_rf = augment_images(X_train, y_train, times=2)  # adjust times as needed
    cfg = Config(use_hog=True, use_lmk=False, aug_for_rf=True, smote=False, model="rf",
                 hog_params=hog_params, shape_predictor_path=shape_predictor_path,
                 rf_params=dict(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1))
    pipe = build_pipeline(cfg)
    print("\n[1] RF + HOG + augmentation (train set augmented)")
    pipe1, f1_1 = train_and_eval(pipe, Xtr_rf, ytr_rf, X_val, y_val, "RF")
    results["1_RF_HOG_AUG"] = (pipe1, f1_1)

    # 2) HOG (RF + SVM + MLP)
    for model in ["rf","svm","mlp"]:
        cfg = Config(use_hog=True, use_lmk=False, model=model, hog_params=hog_params, shape_predictor_path=shape_predictor_path)
        pipe = build_pipeline(cfg)
        print(f"\n[2] {model.upper()} + HOG")
        use_w = (model=="mlp")
        pipe2, f1_2 = train_and_eval(pipe, X_train, y_train, X_val, y_val, model.upper(), use_mlp_weights=use_w)
        results[f"2_{model.upper()}_HOG"] = (pipe2, f1_2)

    # 3) Landmark (RF + SVM + MLP)
    for model in ["rf","svm","mlp"]:
        cfg = Config(use_hog=False, use_lmk=True, model=model, shape_predictor_path=shape_predictor_path)
        pipe = build_pipeline(cfg)
        print(f"\n[3] {model.upper()} + LMK")
        use_w = (model=="mlp")
        pipe3, f1_3 = train_and_eval(pipe, X_train, y_train, X_val, y_val, model.upper(), use_mlp_weights=use_w)
        results[f"3_{model.upper()}_LMK"] = (pipe3, f1_3)

    # 4) Combine HOG + Landmark
    for model in ["rf","svm","mlp"]:
        cfg = Config(use_hog=True, use_lmk=True, model=model, hog_params=hog_params, shape_predictor_path=shape_predictor_path)
        pipe = build_pipeline(cfg)
        print(f"\n[4] {model.upper()} + HOG+LMK")
        use_w = (model=="mlp")
        pipe4, f1_4 = train_and_eval(pipe, X_train, y_train, X_val, y_val, model.upper(), use_mlp_weights=use_w)
        results[f"4_{model.upper()}_HOG_LMK"] = (pipe4, f1_4)

    # 5) Class weight (MLP) -> implemented via sample_weight
    cfg = Config(use_hog=True, use_lmk=True, model="mlp", hog_params=hog_params, shape_predictor_path=shape_predictor_path)
    pipe = build_pipeline(cfg)
    print("\n[5] MLP + class-balanced sample_weight")
    pipe5, f1_5 = train_and_eval(pipe, X_train, y_train, X_val, y_val, "MLP_w", use_mlp_weights=True, classes=np.unique(y_train))
    results["5_MLP_BALANCED"] = (pipe5, f1_5)

    # 6) SMOTE (RF + SVM)
    for model in ["rf","svm"]:
        cfg = Config(use_hog=True, use_lmk=True, model=model, smote=True, hog_params=hog_params, shape_predictor_path=shape_predictor_path)
        pipe = build_pipeline(cfg)
        print(f"\n[6] {model.upper()} + SMOTE + HOG+LMK")
        pipe6, f1_6 = train_and_eval(pipe, X_train, y_train, X_val, y_val, model.upper())
        results[f"6_{model.upper()}_SMOTE"] = (pipe6, f1_6)

    # 7) Standard pipeline knobs already applied globally (scaler + var + corr)
    #    Example with thresholds tightened:
    cfg = Config(use_hog=True, use_lmk=True, model="svm", hog_params=hog_params,
                 shape_predictor_path=shape_predictor_path, var_thresh=1e-5, corr_thresh=0.98,
                 svm_params=dict(C=2.0, gamma=1e-2, kernel="rbf", class_weight=None, random_state=42))
    pipe = build_pipeline(cfg)
    print("\n[7] SVM + Standard pipeline (scaler + low-var + high-corr)")
    pipe7, f1_7 = train_and_eval(pipe, X_train, y_train, X_val, y_val, "SVM_STD")
    results["7_SVM_STDPIPE"] = (pipe7, f1_7)

    return results

# -----------------------
# 11) Save best model (single .joblib with full pipeline)
# -----------------------
import joblib
def save_best(results: Dict[str, Tuple[Any,float]], out_path:str):
    best_key = max(results.keys(), key=lambda k: results[k][1])
    best_pipe, best_f1 = results[best_key]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(dict(pipeline=best_pipe, tag=best_key, f1=best_f1), out_path)
    print(f"[SAVED] {best_key} F1={best_f1:.4f} -> {out_path}")

# -----------------------
# Example usage (fill paths and arrays from your split step)
# -----------------------
if __name__ == "__main__":
    # You already prepared X_train, X_val, y_train, y_val outside.
    # Import them here or place this script where those vars exist.
    # from your_split_module import X_train, y_train, X_val, y_val

    SHAPE_PREDICTOR = "models/shape_predictor_68_face_landmarks.dat"
    assert os.path.exists(SHAPE_PREDICTOR), "Download dlib 68 landmarks model and set the path."

    # results = run_recipes(X_train, y_train, X_val, y_val, SHAPE_PREDICTOR)
    # save_best(results, "models/best_pipeline.joblib")
    pass