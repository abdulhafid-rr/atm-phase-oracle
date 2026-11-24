"""
Bagian lanjutan ATM:
 A) Pemetaan fase PR (paper sec 3.3, Gambar 2) + uji Mann-Whitney per fase (RQ2)
 B) Koneksi Oracle 23ai: simpan fitur sbg VECTOR, cari PR non-assisted termirip (VECTOR_DISTANCE)
Pakai: python atm_phase_oracle.py demo
"""
import os, sys, re
import numpy as np, pandas as pd
import requests
from scipy.stats import mannwhitneyu

# ================= A. FASE PR =================
def fetch_timeline(repo, num, token=os.getenv("GITHUB_TOKEN", "")):
    """Ambil event PR: commit, review, comment, merged (REST timeline API)."""
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/issues/{num}/timeline?per_page=100", headers=h)
    r.raise_for_status()
    return r.json()

def phases_from_events(ev, pr_author):
    """
    ev: DataFrame kolom [t (datetime), type ('commit','review','comment','approved','merged'), actor]
    Return jam per fase: submission, review, waiting, change (definisi paper).
    Fase 5 (setelah acceptance -> merge) dikecualikan karena paper tidak menemukan GPT di situ.
    """
    ev = ev.sort_values("t").reset_index(drop=True)
    commits = ev[ev.type == "commit"]
    others = ev[(ev.type.isin(["review", "comment"])) & (ev.actor != pr_author)]
    appr = ev[ev.type == "approved"]
    if commits.empty or others.empty or appr.empty: return None
    t_first_commit = commits.t.iloc[0]
    t_pull = ev[ev.type == "pull"].t.iloc[0] if (ev.type == "pull").any() else commits.t.iloc[-1]
    t_r1, t_rn, t_acc = others.t.iloc[0], others.t.iloc[-1], appr.t.iloc[0]
    if not (t_first_commit <= t_pull <= t_r1 <= t_rn <= t_acc): return None   # hanya timestamp berurutan
    h = lambda a, b: (b - a).total_seconds() / 3600
    t_r2 = others.t.iloc[1] if len(others) > 1 else t_r1
    t_rn1 = others.t.iloc[-2] if len(others) > 2 else t_r2
    return {"at_submission": h(t_first_commit, t_pull), "at_review": h(t_pull, t_r1),
            "at_waiting": h(t_r2, t_rn1), "at_change": h(t_rn, t_acc)}

def compare_phases(df):
    """df: kolom is_gpt_assisted + 4 kolom fase. Mann-Whitney + median + % reduksi + effect size."""
    rows = []
    for ph in ["at_submission", "at_review", "at_waiting", "at_change"]:
        a = df[df.is_gpt_assisted == 1][ph].dropna(); n = df[df.is_gpt_assisted == 0][ph].dropna()
        if len(a) < 5 or len(n) < 5: continue
        u, p = mannwhitneyu(a, n, alternative="two-sided")
        ma, mn = a.median(), n.median()
        rows.append({"phase": ph, "med_assisted": ma, "med_non": mn,
                     "reduction_%": round(100 * (1 - ma / mn), 1) if mn else np.nan,
                     "U": u, "p": p, "rank_biserial": round(1 - 2 * u / (len(a) * len(n)), 3)})
    out = pd.DataFrame(rows)
    # koreksi Bonferroni: 4 uji berganda -> alpha = 0.05/4 (paper tidak melakukannya; nilai tambah)
    out["signif_bonf"] = out.p < 0.05 / max(len(out), 1)
    return out

# ================= B. ORACLE 23ai =================
def oracle_pipeline(df, user, pwd, dsn):
    """df: kolom pr_id, is_gpt_assisted, time_to_merge_h, commits, pr_size, changed_files, project_age_m"""
    import oracledb, array
    feats = ["commits", "pr_size", "changed_files", "project_age_m"]
    X = df[feats].astype(float).values
    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)          # min-max seperti paper
    con = oracledb.connect(user=user, password=pwd, dsn=dsn); cur = con.cursor()
    cur.execute("BEGIN EXECUTE IMMEDIATE 'DROP TABLE pr_vec PURGE'; EXCEPTION WHEN OTHERS THEN NULL; END;")
    cur.execute("""CREATE TABLE pr_vec (pr_id NUMBER PRIMARY KEY, is_assisted NUMBER,
                   ttm_h NUMBER, feat VECTOR(4, FLOAT32))""")
    cur.executemany("INSERT INTO pr_vec VALUES (:1,:2,:3,:4)",
        [(int(i), int(a), float(t), array.array("f", v)) for i, a, t, v in
         zip(df.pr_id, df.is_gpt_assisted, df.time_to_merge_h, Xn)])
    con.commit()
    # Vector index (materi OA: Vector Indexes) - HNSW in-memory, perlu vector_memory_size > 0
    try:
        cur.execute("""CREATE VECTOR INDEX pr_vec_idx ON pr_vec(feat)
                       ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE MANHATTAN
                       WITH TARGET ACCURACY 95""")
    except Exception as e:
        print("Index dilewati (exact search dipakai):", e)
    matches = []
    for pid, v in zip(df[df.is_gpt_assisted == 1].pr_id, Xn[df.is_gpt_assisted.values == 1]):
        cur.execute("""SELECT pr_id, ttm_h, VECTOR_DISTANCE(feat, :q, MANHATTAN) d
                       FROM pr_vec WHERE is_assisted = 0
                       ORDER BY d FETCH FIRST 1 ROWS ONLY""", q=array.array("f", v))
        r = cur.fetchone()
        if r: matches.append((int(pid), r[0], r[1], r[2]))
    con.close()
    return pd.DataFrame(matches, columns=["assisted_id", "matched_id", "matched_ttm_h", "manhattan_dist"])

# ================= DEMO =================
def demo():
    rng = np.random.default_rng(7); rows = []
    for i in range(300):
        a = i % 2
        rows.append({"is_gpt_assisted": a, "at_submission": rng.exponential(5),
                     "at_review": rng.exponential(1 if a else 3),
                     "at_waiting": rng.exponential(3 if a else 24),
                     "at_change": rng.exponential(1)})
    print(compare_phases(pd.DataFrame(rows)).round(4).to_string(index=False))
    t = pd.Timestamp("2024-01-01")
    ev = pd.DataFrame({"t": [t, t + pd.Timedelta(hours=2), t + pd.Timedelta(hours=3), t + pd.Timedelta(hours=5),
                             t + pd.Timedelta(hours=8), t + pd.Timedelta(hours=9)],
                       "type": ["commit", "pull", "review", "comment", "comment", "approved"],
                       "actor": ["a", "a", "b", "a", "b", "b"]})
    print(phases_from_events(ev, "a"))

if __name__ == "__main__":
    demo()
