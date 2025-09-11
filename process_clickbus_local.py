import os, zipfile, sys, glob
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "desafio clickbus.zip"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "output_mvp"
DO_CLUSTERS = True          
DO_TOP3_ROUTES = True       
SAMPLE_TRAIN  = 400_000     
RANDOM_STATE  = 42

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] ZIP de entrada: {ZIP_PATH}")
print(f"[INFO] Saída em: {OUT_DIR}")


def extract_all(zpath, target_dir):
    with zipfile.ZipFile(zpath, 'r') as z:
        z.extractall(target_dir)

extract_dir = os.path.join(OUT_DIR, "extracted")
os.makedirs(extract_dir, exist_ok=True)
extract_all(ZIP_PATH, extract_dir)


for root, _, files in os.walk(extract_dir):
    for f in files:
        if f.lower().endswith(".zip"):
            inner = os.path.join(root, f)
            inner_out = os.path.join(root, "unzipped_"+os.path.splitext(f)[0])
            os.makedirs(inner_out, exist_ok=True)
            extract_all(inner, inner_out)


candidates = [p for p in glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
              if os.path.basename(p).lower() in ("df_t.csv","df_t .csv","df_t (1).csv")]
if not candidates:
    candidates = [p for p in glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)]
    candidates = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)

assert candidates, "Não encontrei CSVs dentro do zip. Confirme o conteúdo."
DF_T_PATH = candidates[0]
print(f"[INFO] CSV de transações detectado: {DF_T_PATH}")


df = pd.read_csv(DF_T_PATH, low_memory=False)


cols = {c.lower(): c for c in df.columns}
def pick(*names):
    for n in names:
        if n in df.columns: return n
        if n.lower() in cols: return cols[n.lower()]
    return None

col_id  = pick("fk_contact","id_cliente","cliente_id","customer_id")
col_date= pick("date_purchase","dt_compra","data_compra")
col_time= pick("time_purchase","hora_compra","time")
col_val = pick("gmv_success","valor","amount","preco_total")
col_qtd = pick("total_tickets_quantity_success","qtd_tickets","quantidade","tickets")
col_org = pick("place_origin_departure","origem","origin")
col_dst = pick("place_destination_departure","destino","destination")

required = [col_id, col_date, col_val, col_org, col_dst]
assert all(required), f"Colunas essenciais ausentes. Achei: id={col_id}, date={col_date}, valor={col_val}, origem={col_org}, destino={col_dst}"


df[col_val] = pd.to_numeric(df[col_val], errors="coerce")
if col_qtd and col_qtd in df.columns:
    df[col_qtd] = pd.to_numeric(df[col_qtd], errors="coerce")


if col_time and col_time in df.columns:
    try:
        t = pd.to_timedelta(df[col_time].astype(str), errors="coerce")
        dt = pd.to_datetime(df[col_date], errors="coerce") + t
    except:
        dt = pd.to_datetime(df[col_date], errors="coerce")
else:
    dt = pd.to_datetime(df[col_date], errors="coerce")

df["purchase_dt"] = dt
df["trecho"] = df[col_org].astype(str).fillna("NA") + "→" + df[col_dst].astype(str).fillna("NA")


df = df.dropna(subset=[col_id, "purchase_dt"]).copy()
df = df.sort_values([col_id, "purchase_dt"]).reset_index(drop=True)


df["next_dt"] = df.groupby(col_id)["purchase_dt"].shift(-1)
df["delta_dias"] = (df["next_dt"] - df["purchase_dt"]).dt.days
df["label_7d"] = ((df["delta_dias"].notna()) & (df["delta_dias"] <= 7)).astype(int)

labels_path = os.path.join(OUT_DIR, "labels_clientes.csv")
df.loc[df["next_dt"].notna(), [col_id,"purchase_dt","next_dt","delta_dias","label_7d"]].to_csv(labels_path, index=False)
print(f"[OK] Labels salvos: {labels_path}")


base = df.loc[df["next_dt"].notna()].copy()


ref_dt = df["purchase_dt"].max()
lookback = ref_dt - pd.Timedelta(days=365)
hist = df.loc[df["purchase_dt"] >= lookback].copy()

R = (ref_dt - hist.groupby(col_id)["purchase_dt"].max()).dt.days.rename("recency_dias")
F = hist.groupby(col_id).size().rename("frequencia_12m")
M = hist.groupby(col_id)[col_val].mean().rename("ticket_medio_12m")
VT= hist.groupby(col_id)[col_val].sum().rename("valor_total_12m")
VR= hist.groupby(col_id)["trecho"].nunique().rename("variedade_rotas")

feat = pd.concat([R,F,M,VT,VR], axis=1).reset_index().rename(columns={col_id:"id_cliente"})
first_purchase = df.groupby(col_id)["purchase_dt"].min().rename("primeira_compra")
last_purchase  = df.groupby(col_id)["purchase_dt"].max().rename("ultima_compra")
life_days = (last_purchase - first_purchase).dt.days.rename("dias_de_vida")
feat = feat.merge(first_purchase, left_on="id_cliente", right_index=True, how="left")\
           .merge(last_purchase,  left_on="id_cliente", right_index=True, how="left")\
           .merge(life_days,      left_on="id_cliente", right_index=True, how="left")
feat_path = os.path.join(OUT_DIR, "stage_features_base.csv")
feat.to_csv(feat_path, index=False)
print(f"[OK] Features salvas: {feat_path} (linhas: {len(feat)})")


seg_cols = ["recency_dias","frequencia_12m","ticket_medio_12m","valor_total_12m","variedade_rotas","dias_de_vida"]
train = base.merge(feat, left_on=col_id, right_on="id_cliente", how="left")


if len(train) > SAMPLE_TRAIN:
    train = train.sample(SAMPLE_TRAIN, random_state=RANDOM_STATE)

X = train[seg_cols].fillna(0)
y = train["label_7d"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(Xtr, ytr)
proba = rf.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte, proba)
report = classification_report(yte, (proba>0.5).astype(int), digits=3)
print("\n[METRICAS RF 7d]\nAUC:", round(auc,4))
print(report)


last_tx = df.sort_values([col_id,"purchase_dt"]).groupby(col_id).tail(1)
scoring = last_tx[[col_id,"purchase_dt"]].merge(feat, left_on=col_id, right_on="id_cliente", how="left").fillna(0)
X_score = scoring[seg_cols]
scoring["score_recompra_7d_RF"] = rf.predict_proba(X_score)[:,1]
scoring = scoring.rename(columns={"purchase_dt":"ultima_compra"})
pred_path = os.path.join(OUT_DIR, "previsoes_7d_RF.csv")
scoring[["id_cliente","ultima_compra","score_recompra_7d_RF"]].to_csv(pred_path, index=False)
print(f"[OK] Previsões salvas: {pred_path} (linhas: {len(scoring)})")


if DO_CLUSTERS:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(feat[seg_cols].fillna(0).values)
    mbk = MiniBatchKMeans(n_clusters=4, random_state=RANDOM_STATE, batch_size=2048, n_init=5, max_no_improvement=10)
    feat["cluster_id"] = mbk.fit_predict(Xz)
    clusters_path = os.path.join(OUT_DIR, "clusters.csv")
    feat[["id_cliente","cluster_id"] + seg_cols].to_csv(clusters_path, index=False)
    print(f"[OK] Clusters salvos: {clusters_path}")


if DO_TOP3_ROUTES:
    hist = df[[col_id,"trecho"]].copy()
    vc = hist.value_counts().reset_index(name="freq").sort_values([col_id,"freq"], ascending=[True, False])

    top3_list = []
    current_id, buffer = None, []
    for cid, tr, freq in vc.itertuples(index=False):
        if cid != current_id:
            if current_id is not None:
                top3_list.append((current_id, '; '.join(buffer)))
            current_id, buffer = cid, [tr]
        else:
            if len(buffer) < 3:
                buffer.append(tr)
    if current_id is not None:
        top3_list.append((current_id, '; '.join(buffer)))

    top3_df = pd.DataFrame(top3_list, columns=["id_cliente","sugestoes_top3"])
    top3_path = os.path.join(OUT_DIR, "next_route_topk.csv")
    top3_df.to_csv(top3_path, index=False)
    print(f"[OK] Top-3 rotas salvas: {top3_path}")

print("\n[FINALIZADO] Arquivos para o Power BI gerados em:", OUT_DIR)
