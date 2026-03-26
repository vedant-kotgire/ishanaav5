import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import io
import os
import pathlib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Ishanaa Analytics",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BULLETPROOF DATA PATH RESOLUTION
# ============================================================
def find_data_dir():
    """Try multiple strategies to locate the data/ folder."""
    candidates = []

    # Strategy 1: relative to this script file
    try:
        script_dir = pathlib.Path(__file__).resolve().parent
        candidates.append(script_dir / 'data')
    except Exception:
        pass

    # Strategy 2: current working directory
    candidates.append(pathlib.Path.cwd() / 'data')

    # Strategy 3: Streamlit Cloud mount paths
    candidates.append(pathlib.Path('/mount/src') / 'vk-ishanaa' / 'data')
    candidates.append(pathlib.Path('/mount/src') / 'ishanaa' / 'data')

    # Strategy 4: walk up from cwd
    cwd = pathlib.Path.cwd()
    for parent in [cwd, cwd.parent, cwd.parent.parent]:
        candidates.append(parent / 'data')

    # Strategy 5: search for the CSV file anywhere common
    for p in candidates:
        if p.is_dir() and (p / 'ishanaa_survey_raw.csv').is_file():
            return str(p)

    # Strategy 6: glob search from /mount/src if on Streamlit Cloud
    mount_src = pathlib.Path('/mount/src')
    if mount_src.is_dir():
        found = list(mount_src.rglob('ishanaa_survey_raw.csv'))
        if found:
            return str(found[0].parent)

    # Fallback: return relative 'data' and let it fail with a clear message
    return 'data'

DATA_DIR = find_data_dir()

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.insight-box {
    background: linear-gradient(135deg, #f4ece6, #fdf5ef);
    border-left: 3px solid #c25e3a;
    padding: 16px 20px;
    border-radius: 0 10px 10px 0;
    margin: 16px 0;
}
.insight-box strong { color: #c25e3a; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stMetric"] {
    background: white; border: 1px solid #e8e2dc; border-radius: 12px; padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    try:
        raw = pd.read_csv(os.path.join(DATA_DIR, 'ishanaa_survey_raw.csv'))
        enc = pd.read_csv(os.path.join(DATA_DIR, 'ishanaa_survey_encoded.csv'))
        ddict = pd.read_csv(os.path.join(DATA_DIR, 'ishanaa_data_dictionary.csv'))
        return raw, enc, ddict
    except FileNotFoundError as e:
        st.error(f"Could not find data files. Searched in: {DATA_DIR}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Script location: {os.path.abspath(__file__) if '__file__' in dir() else 'unknown'}")
        # List what IS in the directory
        try:
            cwd_contents = os.listdir(os.getcwd())
            st.error(f"CWD contents: {cwd_contents}")
            if os.path.isdir('data'):
                st.error(f"data/ contents: {os.listdir('data')}")
        except Exception:
            pass
        st.stop()

df_raw, df_enc, df_dict = load_data()

# ============================================================
# HELPERS
# ============================================================
def insight_box(title, text):
    st.markdown(f'<div class="insight-box"><strong>{title}</strong><br>{text}</div>', unsafe_allow_html=True)

def expand_multi_col(series, sep='; '):
    items = []
    for val in series.dropna():
        items.extend([v.strip() for v in str(val).split(sep) if v.strip()])
    return pd.Series(items).value_counts() if items else pd.Series(dtype=int)

def get_feature_cols():
    skip = {'respondent_id', 'persona', 'purchase_intent_ordinal',
            'purchase_intent_binary', 'spend_per_kurti_ordinal',
            'spend_per_kurti_midpoint_AED', 'cluster'}
    return [c for c in df_enc.columns if c not in skip]

@st.cache_data
def get_feature_matrix():
    cols = get_feature_cols()
    X = df_enc[cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=cols, index=X.index)
    return X, cols

@st.cache_data
def get_clf_data():
    X, cols = get_feature_matrix()
    y = df_enc['purchase_intent_binary'].copy()
    mask = y.notna()
    return X.loc[mask].reset_index(drop=True), y[mask].astype(int).reset_index(drop=True), cols

@st.cache_data
def get_reg_data():
    X, cols = get_feature_matrix()
    y = df_enc['spend_per_kurti_midpoint_AED'].copy()
    mask = y.notna()
    return X.loc[mask].reset_index(drop=True), y[mask].reset_index(drop=True), cols

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 👗 **Ishanaa Analytics**")
    st.markdown("*Data-Driven Decision Making*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Descriptive Analysis",
        "🔍 Diagnostic Analysis",
        "🔮 Predictive Analysis",
        "💡 Prescriptive Analysis",
        "🆕 New Customer Predictor",
        "ℹ️ Data Dictionary"
    ], index=0)
    st.markdown("---")
    st.caption(f"Dataset: {len(df_raw):,} respondents · {df_enc.shape[1]} features")
    st.caption("UAE Market · March 2026")


# ================================================================
#  PAGE 1: DESCRIPTIVE
# ================================================================
if page == "📊 Descriptive Analysis":
    st.title("📊 Descriptive Analytics")
    st.markdown("*What does our potential UAE market look like?*")

    valid_intent = df_enc['purchase_intent_binary'].dropna()
    interested_pct = (valid_intent.mean() * 100) if len(valid_intent) > 0 else 0
    avg_spend = df_enc['spend_per_kurti_midpoint_AED'].mean()
    pack_series = expand_multi_col(df_raw['Q19_Pack_Interest'])
    top_pack = pack_series.index[0] if len(pack_series) > 0 else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Respondents", f"{len(df_raw):,}")
    c2.metric("Interested (%)", f"{interested_pct:.1f}%")
    c3.metric("Avg Spend / Kurti", f"{avg_spend:.0f} AED")
    c4.metric("Top Pack", (top_pack[:25] + "...") if len(str(top_pack)) > 25 else top_pack)

    st.markdown("---")
    st.subheader("1.1 — Demographic Profile")
    dc1, dc2 = st.columns(2)

    with dc1:
        age_o = ["Under 18", "18-22", "23-27", "28-34", "35 and above"]
        ac = df_raw['Q1_Age'].value_counts().reindex(age_o).dropna()
        fig = px.bar(x=ac.index, y=ac.values, color_discrete_sequence=['#c25e3a'],
                     labels={'x': 'Age Group', 'y': 'Count'}, title="Age Distribution")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with dc2:
        oc = df_raw['Q2_Occupation'].value_counts()
        fig = px.pie(values=oc.values, names=oc.index, title="Occupation",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    dc3, dc4 = st.columns(2)
    with dc3:
        inc_o = ["No personal income", "Under 3,000 AED", "3,000-7,000 AED",
                 "7,001-12,000 AED", "12,001-20,000 AED", "Above 20,000 AED"]
        ic = df_raw['Q4_Income'].value_counts().reindex(inc_o).dropna()
        fig = px.bar(x=ic.index, y=ic.values, color_discrete_sequence=['#3a6b8c'],
                     labels={'x': 'Income Band', 'y': 'Count'}, title="Income (AED)")
        fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with dc4:
        ec = df_raw['Q3_Ethnicity'].value_counts()
        fig = px.pie(values=ec.values, names=ec.index, title="Ethnicity",
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("1.2 — Shopping Behavior")
    sb1, sb2 = st.columns(2)

    with sb1:
        sp_o = ["Under 30 AED", "30-60 AED", "61-100 AED", "101-150 AED", "151-250 AED", "Above 250 AED"]
        sc = df_raw['Q9_Spend_Per_Kurti'].value_counts().reindex(sp_o).dropna()
        fig = px.bar(x=sc.index, y=sc.values, color_discrete_sequence=['#3a7d5c'],
                     labels={'x': 'Spend', 'y': 'Count'}, title="Spend Per Kurti (AED)")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with sb2:
        bc = expand_multi_col(df_raw['Q7_Current_Brands']).head(10)
        if len(bc) > 0:
            fig = px.bar(x=bc.values, y=bc.index, color_discrete_sequence=['#7a5195'],
                          labels={'x': 'Selections', 'y': ''}, title="Brand Landscape", orientation='h')
            fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("1.3 — Product Preferences")
    pp1, pp2, pp3 = st.columns(3)

    for col_data, container, title, color in [
        ('Q11_Style_Preference', pp1, "Styles", '#c25e3a'),
        ('Q12_Fabric_Preference', pp2, "Fabrics", '#3a6b8c'),
        ('Q13_Color_Preference', pp3, "Colors", '#3a7d5c')
    ]:
        with container:
            vc = expand_multi_col(df_raw[col_data])
            if len(vc) > 0:
                fig = px.bar(x=vc.values, y=vc.index, color_discrete_sequence=[color],
                              labels={'x': 'Selections', 'y': ''}, title=title, orientation='h')
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("1.4 — Purchase Intent & Pain Points")
    pi1, pi2 = st.columns(2)

    with pi1:
        io = ["Definitely would buy", "Probably would buy", "Might or might not",
              "Probably would not", "Definitely would not"]
        ivc = df_raw['Q25_Purchase_Intent'].value_counts().reindex(io).dropna()
        cmap = {'Definitely would buy': '#3a7d5c', 'Probably would buy': '#5aa87a',
                'Might or might not': '#b8943e', 'Probably would not': '#e07050',
                'Definitely would not': '#c0392b'}
        fig = px.bar(x=ivc.values, y=ivc.index, orientation='h', color=ivc.index,
                     color_discrete_map=cmap, labels={'x': 'Count', 'y': ''}, title="Purchase Intent")
        fig.update_layout(height=350, showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with pi2:
        pnc = expand_multi_col(df_raw['Q23_Pain_Points']).head(9)
        if len(pnc) > 0:
            fig = px.bar(x=pnc.values, y=pnc.index, color_discrete_sequence=['#c0392b'],
                          labels={'x': 'Selections', 'y': ''}, title="Pain Points", orientation='h')
            fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("1.5 — Pack Interest")
    if len(pack_series) > 0:
        fig = px.bar(x=pack_series.index, y=pack_series.values, color_discrete_sequence=['#7a5195'],
                     labels={'x': 'Pack', 'y': 'Selections'}, title="Ishanaa Pack Interest")
        fig.update_layout(height=380, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)
    insight_box("So What?", f"Top pack: <b>{top_pack}</b>. {interested_pct:.0f}% show purchase interest — strong UAE market validation.")


# ================================================================
#  PAGE 2: DIAGNOSTIC
# ================================================================
elif page == "🔍 Diagnostic Analysis":
    st.title("🔍 Diagnostic Analytics")
    st.markdown("*Why do certain customers want to buy — and others don't?*")

    tab1, tab2, tab3 = st.tabs(["🧩 Clustering", "🔗 Association Rules", "📈 Correlations"])

    with tab1:
        st.subheader("2.1 — Customer Persona Discovery")
        X_cl, _ = get_feature_matrix()
        scl = StandardScaler()
        Xs = scl.fit_transform(X_cl)

        el1, el2 = st.columns(2)
        K_range = range(2, 9)
        inertias, sils = [], []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lb = km.fit_predict(Xs)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(Xs, lb, sample_size=min(800, len(Xs))))

        with el1:
            fig = px.line(x=list(K_range), y=inertias, markers=True, labels={'x': 'K', 'y': 'Inertia'}, title="Elbow Method")
            fig.update_traces(line_color='#c25e3a'); fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with el2:
            fig = px.line(x=list(K_range), y=sils, markers=True, labels={'x': 'K', 'y': 'Silhouette'}, title="Silhouette Scores")
            fig.update_traces(line_color='#3a7d5c'); fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        k_choice = st.slider("Select K", 2, 8, list(K_range)[int(np.argmax(sils))])
        km_f = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
        clabs = km_f.fit_predict(Xs)

        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        fig = px.scatter(x=Xp[:, 0], y=Xp[:, 1], color=[str(c) for c in clabs],
                         labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                         title=f"K-Means (k={k_choice})", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Silhouette: {silhouette_score(Xs, clabs, sample_size=min(800, len(Xs))):.3f}**")

        # Profiles
        df_t = df_enc.copy(); df_t['cluster'] = clabs
        pc = ['age_ordinal', 'income_midpoint_AED', 'wardrobe_count', 'shopping_freq_ordinal',
              'spend_per_kurti_midpoint_AED', 'bundle_budget_midpoint_AED', 'return_anxiety_ordinal']
        sc2 = [c for c in pc if c in df_t.columns]
        if sc2:
            cp = df_t.groupby('cluster')[sc2].mean().round(1)
            cp['size'] = df_t.groupby('cluster').size()
            if 'purchase_intent_binary' in df_t.columns:
                cp['interest_%'] = (df_t.groupby('cluster')['purchase_intent_binary'].mean() * 100).round(1)
            st.dataframe(cp.style.background_gradient(cmap='YlOrRd'), use_container_width=True)

            # Radar
            rn = MinMaxScaler().fit_transform(cp[sc2])
            rd = pd.DataFrame(rn, columns=sc2, index=cp.index)
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, cl in enumerate(rd.index):
                v = rd.loc[cl].tolist() + [rd.loc[cl].tolist()[0]]
                c = sc2 + [sc2[0]]
                fig.add_trace(go.Scatterpolar(r=v, theta=c, name=f'Cluster {cl}', line_color=colors[i % len(colors)]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Radar", height=450)
            st.plotly_chart(fig, use_container_width=True)

        # DBSCAN
        st.markdown("---")
        st.subheader("2.2 — DBSCAN")
        db = DBSCAN(eps=3.5, min_samples=8)
        dl = db.fit_predict(Xs)
        ndc = len(set(dl)) - (1 if -1 in dl else 0)
        nn = int((dl == -1).sum())
        fig = px.scatter(x=Xp[:, 0], y=Xp[:, 1], color=[str(c) for c in dl],
                         labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                         title=f"DBSCAN: {ndc} clusters, {nn} noise", color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Dendrogram
        st.subheader("2.3 — Dendrogram")
        np.random.seed(42)
        si = np.random.choice(len(Xs), size=min(200, len(Xs)), replace=False)
        Z = linkage(Xs[si], method='ward')
        fg, ax = plt.subplots(figsize=(12, 4))
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8, ax=ax, color_threshold=50)
        ax.set_title("Hierarchical Dendrogram (Ward)"); ax.set_ylabel("Distance")
        plt.tight_layout(); st.pyplot(fg); plt.close(fg)

    with tab2:
        st.subheader("2.4 — Association Rules")
        if not HAS_MLXTEND:
            st.error("mlxtend not installed.")
        else:
            arm_type = st.selectbox("Basket Type", [
                "Style × Fabric × Color", "Sleeve × Neckline × Length",
                "Occasion × Pack × Discount", "Pain × Channel × Brand"])
            pmap = {"Style": ('style_', 'fabric_', 'color_'), "Sleeve": ('sleeve_', 'neck_', 'length_'),
                    "Occasion": ('occasion_', 'pack_', 'discount_'), "Pain": ('pain_', 'channel_', 'brand_')}
            pfx = pmap[arm_type.split(' ')[0]]
            cols = [c for c in df_enc.columns if c.startswith(pfx)]
            if cols:
                basket = df_enc[cols].fillna(0).astype(bool)
                basket = basket.loc[:, basket.sum() > 20]
                min_sup = st.slider("Min Support", 0.05, 0.40, 0.10, 0.01)
                if len(basket.columns) >= 2:
                    try:
                        freq = apriori(basket, min_support=min_sup, use_colnames=True)
                        if len(freq) > 1:
                            try:
                                rules = association_rules(freq, metric="lift", min_threshold=1.1)
                            except TypeError:
                                rules = association_rules(freq, metric="lift", min_threshold=1.1, num_itemsets=len(freq))
                            if len(rules) > 0:
                                rules['IF'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                                rules['THEN'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
                                rt = rules.sort_values('lift', ascending=False).head(25)
                                st.markdown(f"**{len(rules)} rules found** (top 25)")
                                st.dataframe(rt[['IF', 'THEN', 'support', 'confidence', 'lift']].style.format(
                                    {'support': '{:.3f}', 'confidence': '{:.3f}', 'lift': '{:.2f}'}).background_gradient(
                                    subset=['lift'], cmap='YlOrRd'), use_container_width=True, height=500)
                                if len(rt) >= 3:
                                    fig = px.scatter(rt, x='support', y='confidence', size='lift', color='lift',
                                                     hover_data=['IF', 'THEN'], title="Support vs Confidence",
                                                     color_continuous_scale='YlOrRd')
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                            else: st.warning("No rules at this threshold.")
                        else: st.warning("Not enough frequent itemsets. Lower support.")
                    except Exception as e:
                        st.error(f"ARM error: {e}")
                else: st.warning("Not enough columns for this basket.")

    with tab3:
        st.subheader("2.5 — Correlations & Feature Importance")
        nc = [c for c in ['age_ordinal', 'income_midpoint_AED', 'wardrobe_count', 'shopping_freq_ordinal',
              'spend_per_kurti_midpoint_AED', 'bundle_budget_midpoint_AED', 'return_anxiety_ordinal',
              'purchase_intent_ordinal'] if c in df_enc.columns]
        if nc:
            fig = px.imshow(df_enc[nc].corr(), text_auto='.2f', color_continuous_scale='RdBu_r',
                            aspect='auto', title="Correlation Heatmap")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Feature Importance (RF → Purchase Intent)**")
        Xc, yc, _ = get_clf_data()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(Xc, yc)
        imp = pd.Series(rf.feature_importances_, index=Xc.columns).sort_values(ascending=False).head(20)
        fig = px.bar(x=imp.values, y=imp.index, color_discrete_sequence=['#c25e3a'],
                      labels={'x': 'Importance', 'y': ''}, title="Top 20 Features", orientation='h')
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
#  PAGE 3: PREDICTIVE
# ================================================================
elif page == "🔮 Predictive Analysis":
    st.title("🔮 Predictive Analytics")
    tab1, tab2 = st.tabs(["🎯 Classification", "💰 Regression"])

    with tab1:
        st.subheader("3.1 — Purchase Interest Classifier")
        X, y, _ = get_clf_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)

        models = {'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                  'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
                  'SVM': SVC(kernel='rbf', probability=True, random_state=42)}
        if HAS_XGB:
            models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss', verbosity=0)

        res, roc_d = {}, {}
        with st.spinner("Training classifiers..."):
            for nm, m in models.items():
                try:
                    m.fit(Xtrs, ytr); yp = m.predict(Xtes); ypr = m.predict_proba(Xtes)[:, 1]
                    res[nm] = {'Accuracy': accuracy_score(yte, yp), 'Precision': precision_score(yte, yp, zero_division=0),
                               'Recall': recall_score(yte, yp, zero_division=0), 'F1': f1_score(yte, yp, zero_division=0),
                               'AUC': roc_auc_score(yte, ypr)}
                    fpr, tpr, _ = roc_curve(yte, ypr); roc_d[nm] = (fpr, tpr)
                except Exception as e: st.warning(f"{nm}: {e}")

        if res:
            rdf = pd.DataFrame(res).T
            st.dataframe(rdf.style.format('{:.4f}').background_gradient(cmap='Greens', axis=0), use_container_width=True)
            bn = rdf['F1'].idxmax(); bf = rdf.loc[bn, 'F1']
            st.success(f"**Best: {bn}** (F1 = {bf:.4f})")

            mc1, mc2 = st.columns(2)
            with mc1:
                fig = go.Figure()
                clrs = ['#c25e3a', '#3a7d5c', '#7a5195', '#3a6b8c']
                for i, (n, (f, t)) in enumerate(roc_d.items()):
                    fig.add_trace(go.Scatter(x=f, y=t, name=f'{n} ({res[n]["AUC"]:.3f})', line=dict(color=clrs[i % len(clrs)])))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='grey'), showlegend=False))
                fig.update_layout(title="ROC Curves", height=400, xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig, use_container_width=True)
            with mc2:
                bm = models[bn]; cm = confusion_matrix(yte, bm.predict(Xtes))
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Oranges',
                                x=['Not Interested', 'Interested'], y=['Not Interested', 'Interested'], title=f"Confusion — {bn}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            if hasattr(bm, 'feature_importances_'):
                fi = pd.Series(bm.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            elif hasattr(bm, 'coef_'):
                fi = pd.Series(np.abs(bm.coef_[0]), index=X.columns).sort_values(ascending=False).head(15)
            else:
                rft = RandomForestClassifier(100, random_state=42, max_depth=10); rft.fit(Xtrs, ytr)
                fi = pd.Series(rft.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            fig = px.bar(x=fi.values, y=fi.index, color_discrete_sequence=['#c25e3a'], labels={'x': 'Importance', 'y': ''}, title="Top 15 Features", orientation='h')
            fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("3.2 — Spending Predictor")
        Xr, yr, _ = get_reg_data()
        Xrt, Xre, yrt, yre = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        sr = StandardScaler(); Xrts = sr.fit_transform(Xrt); Xres = sr.transform(Xre)

        rms = {'Linear': LinearRegression(), 'Ridge': Ridge(), 'Lasso': Lasso(),
               'Random Forest': RandomForestRegressor(200, max_depth=10, random_state=42),
               'Gradient Boosting': GradientBoostingRegressor(200, max_depth=5, random_state=42)}
        rr = {}
        with st.spinner("Training regressors..."):
            for n, m in rms.items():
                try:
                    m.fit(Xrts, yrt); yp = m.predict(Xres)
                    rr[n] = {'MAE': mean_absolute_error(yre, yp), 'RMSE': np.sqrt(mean_squared_error(yre, yp)), 'R²': r2_score(yre, yp)}
                except Exception as e: st.warning(f"{n}: {e}")
        if rr:
            rrd = pd.DataFrame(rr).T
            st.dataframe(rrd.style.format({'MAE': '{:.1f}', 'RMSE': '{:.1f}', 'R²': '{:.4f}'}).background_gradient(subset=['R²'], cmap='Greens'), use_container_width=True)
            brn = rrd['R²'].idxmax(); br2 = rrd.loc[brn, 'R²']
            st.success(f"**Best: {brn}** (R² = {br2:.4f})")
            brm = rms[brn]; yrp = brm.predict(Xres)
            r1, r2 = st.columns(2)
            with r1:
                fig = px.scatter(x=yre, y=yrp, labels={'x': 'Actual', 'y': 'Predicted'}, title=f"Actual vs Predicted — {brn}", opacity=0.5, color_discrete_sequence=['#3a6b8c'])
                fig.add_trace(go.Scatter(x=[0, 350], y=[0, 350], mode='lines', line=dict(dash='dash', color='red'), name='Perfect'))
                fig.update_layout(height=400); st.plotly_chart(fig, use_container_width=True)
            with r2:
                fig = px.histogram(x=yre.values - yrp, nbins=40, color_discrete_sequence=['#7a5195'], labels={'x': 'Residual', 'y': 'Count'}, title="Residuals")
                fig.update_layout(height=400); st.plotly_chart(fig, use_container_width=True)


# ================================================================
#  PAGE 4: PRESCRIPTIVE
# ================================================================
elif page == "💡 Prescriptive Analysis":
    st.title("💡 Prescriptive Analytics")
    X_cl, _ = get_feature_matrix()
    sc = StandardScaler(); Xs = sc.fit_transform(X_cl)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    cl = km.fit_predict(Xs); dp = df_enc.copy(); dp['cluster'] = cl

    st.subheader("4.1 — Segment Priority Scorecard")
    rows = []
    for c in sorted(dp['cluster'].unique()):
        m = dp['cluster'] == c; rm = m[:len(df_raw)]
        sz = int(m.sum())
        ir = dp.loc[m, 'purchase_intent_binary'].mean(); ir = ir if pd.notna(ir) else 0
        sp = dp.loc[m, 'spend_per_kurti_midpoint_AED'].mean(); sp = sp if pd.notna(sp) else 0
        pr = sz * ir * sp / 1000
        pk = expand_multi_col(df_raw.loc[rm, 'Q19_Pack_Interest'].dropna())
        pn = expand_multi_col(df_raw.loc[rm, 'Q23_Pain_Points'].dropna())
        ds = expand_multi_col(df_raw.loc[rm, 'Q21_Discount_Preference'].dropna())
        rows.append({'Cluster': c, 'Size': sz, 'Interest %': round(ir * 100, 1), 'Avg Spend': round(sp),
                     'Priority': round(pr, 1), 'Top Pack': str(pk.index[0])[:45] if len(pk) > 0 else 'N/A',
                     'Top Pain': str(pn.index[0])[:45] if len(pn) > 0 else 'N/A',
                     'Best Discount': str(ds.index[0])[:45] if len(ds) > 0 else 'N/A'})

    sdf = pd.DataFrame(rows).sort_values('Priority', ascending=False)
    st.dataframe(sdf.style.background_gradient(subset=['Priority'], cmap='YlOrRd').background_gradient(subset=['Interest %'], cmap='Greens'), use_container_width=True)

    t = sdf.iloc[0]; s = sdf.iloc[1] if len(sdf) > 1 else t
    insight_box("Launch Rec", f"<b>Cluster {int(t['Cluster'])}</b> — Priority {t['Priority']}. {int(t['Size'])} people, {t['Interest %']}% interest, ~{int(t['Avg Spend'])} AED. Lead: <b>{t['Top Pack']}</b>. Offer: <b>{t['Best Discount']}</b>.")

    st.markdown("---")
    st.subheader("4.2 — Discount Matrix")
    dd = []
    for c in sorted(dp['cluster'].unique()):
        rm = (dp['cluster'] == c)[:len(df_raw)]
        dc = expand_multi_col(df_raw.loc[rm, 'Q21_Discount_Preference'].dropna())
        for dt, cnt in dc.head(5).items():
            dd.append({'Cluster': f'C{c}', 'Discount': str(dt), 'Count': cnt})
    if dd:
        fig = px.bar(pd.DataFrame(dd), x='Cluster', y='Count', color='Discount', barmode='group',
                     color_discrete_sequence=px.colors.qualitative.Set2, title="Discounts by Cluster")
        fig.update_layout(height=420); st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("4.3 — Launch Playbook")
    st.markdown(f"""
**🚀 Primary:** Cluster {int(t['Cluster'])} ({int(t['Size'])} ppl, {t['Interest %']}%) → **{t['Top Pack']}** at ~{int(t['Avg Spend'])} AED, offer **{t['Best Discount']}**

**🎯 Secondary:** Cluster {int(s['Cluster'])} ({int(s['Size'])} ppl, {s['Interest %']}%) → **{s['Top Pack']}**, offer **{s['Best Discount']}**

**Budget:** 50% primary → 30% secondary → 20% testing
    """)


# ================================================================
#  PAGE 5: NEW CUSTOMER PREDICTOR
# ================================================================
elif page == "🆕 New Customer Predictor":
    st.title("🆕 New Customer Predictor")

    @st.cache_resource
    def train_prod():
        Xc, yc, _ = get_clf_data(); Xr, yr, _ = get_reg_data(); Xa, ac = get_feature_matrix()
        s1 = StandardScaler(); X1 = s1.fit_transform(Xc)
        clf = RandomForestClassifier(200, max_depth=12, random_state=42); clf.fit(X1, yc)
        s2 = StandardScaler(); X2 = s2.fit_transform(Xr)
        reg = GradientBoostingRegressor(200, max_depth=5, random_state=42); reg.fit(X2, yr)
        s3 = StandardScaler(); X3 = s3.fit_transform(Xa)
        km = KMeans(5, random_state=42, n_init=10); km.fit(X3)
        return clf, reg, km, s1, s2, s3, list(Xc.columns), list(Xr.columns), ac, Xa.median()

    clf, reg, km, s1, s2, s3, cc, rc, ac, med = train_prod()

    mode = st.radio("Mode", ["📝 Quick Form", "📤 Bulk CSV"], horizontal=True)

    if mode == "📝 Quick Form":
        f1, f2, f3 = st.columns(3)
        with f1:
            ia = st.selectbox("Age", ["Under 18", "18-22", "23-27", "28-34", "35+"])
            io = st.selectbox("Occupation", ["Student", "Postgrad Student", "Professional", "Self-Employed", "Homemaker", "Other"])
            ie = st.selectbox("Ethnicity", ["South Asian", "Arab/ME", "SE Asian", "African", "Western", "Other"])
        with f2:
            ii = st.selectbox("Income", ["No income", "<3K AED", "3-7K", "7-12K", "12-20K", ">20K"])
            iw = st.selectbox("Kurtis Owned", ["None", "1-5", "6-12", "13-20", "20+"])
            iq = st.selectbox("Frequency", ["Weekly", "2-3x/month", "Monthly", "Every 2-3 months", "Festivals only", "Rarely"])
        with f3:
            ifi = st.selectbox("Fashion ID", ["Trend follower", "Set style", "Comfort first", "Occasion only", "Fusion lover"])
            ira = st.selectbox("Return Anxiety", ["Very worried", "Sometimes", "Rarely", "Never"])

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            am = {"Under 18": 1, "18-22": 2, "23-27": 3, "28-34": 4, "35+": 5}
            im = {"No income": 0, "<3K AED": 1500, "3-7K": 5000, "7-12K": 9500, "12-20K": 16000, ">20K": 25000}
            iom = {"No income": 0, "<3K AED": 1, "3-7K": 2, "7-12K": 3, "12-20K": 4, ">20K": 5}
            wm = {"None": 0, "1-5": 3, "6-12": 9, "13-20": 16, "20+": 25}
            fm = {"Weekly": 6, "2-3x/month": 5, "Monthly": 4, "Every 2-3 months": 3, "Festivals only": 2, "Rarely": 1}
            rm = {"Very worried": 4, "Sometimes": 3, "Rarely": 2, "Never": 1}
            fim = {"Trend follower": 1, "Set style": 2, "Comfort first": 3, "Occasion only": 4, "Fusion lover": 5}

            nr = pd.DataFrame([med], columns=ac)
            for col, val in [('age_ordinal', am.get(ia, 3)), ('income_ordinal', iom.get(ii, 2)),
                             ('income_midpoint_AED', im.get(ii, 5000)), ('wardrobe_count', wm.get(iw, 9)),
                             ('shopping_freq_ordinal', fm.get(iq, 4)), ('return_anxiety_ordinal', rm.get(ira, 2))]:
                if col in nr.columns: nr[col] = val
            fi = fim.get(ifi, 3)
            for i in range(1, 6):
                if f'fashion_id_{i}' in nr.columns: nr[f'fashion_id_{i}'] = 1 if i == fi else 0

            try:
                prob = clf.predict_proba(s1.transform(nr[cc].fillna(0)))[0][1]
            except: prob = 0.5
            try:
                spd = float(reg.predict(s2.transform(nr[rc].fillna(0)))[0])
            except: spd = 80.0
            try:
                cid = int(km.predict(s3.transform(nr[ac].fillna(0)))[0])
            except: cid = 0

            st.markdown("---")
            st.markdown("### 🎯 Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Interest", "✅ Yes" if prob >= 0.5 else "❌ No")
            r2.metric("Probability", f"{prob:.1%}")
            r3.metric("Spend", f"{max(15, spd):.0f} AED")
            r4.metric("Cluster", f"C{cid}")

            if prob >= 0.7: st.success("🎯 **HIGH VALUE** — Show premium packs.")
            elif prob >= 0.5: st.info("👍 **WARM** — Start with Campus Essentials + discount.")
            elif prob >= 0.3: st.warning("🤔 **LUKEWARM** — Retarget with social proof.")
            else: st.error("❌ **LOW** — Save ad spend.")

    else:
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            try:
                nd = pd.read_csv(uploaded)
                st.success(f"{len(nd)} rows × {nd.shape[1]} cols")
                st.dataframe(nd.head(), use_container_width=True)
                if st.button("🔮 Predict All", type="primary"):
                    with st.spinner("Processing..."):
                        for c in ac:
                            if c not in nd.columns: nd[c] = med.get(c, 0)
                        nX = nd[ac].fillna(0)
                        for c in nX.columns: nX[c] = pd.to_numeric(nX[c], errors='coerce').fillna(0)
                        try: pr = clf.predict_proba(s1.transform(nX[cc]))[:, 1]; pd_ = (pr >= 0.5).astype(int)
                        except: pr = np.full(len(nX), 0.5); pd_ = np.zeros(len(nX), dtype=int)
                        try: sp = reg.predict(s2.transform(nX[rc]))
                        except: sp = np.full(len(nX), 80)
                        try: ck = km.predict(s3.transform(nX[ac]))
                        except: ck = np.zeros(len(nX), dtype=int)
                        nd['predicted_interest'] = pd_; nd['interest_prob'] = np.round(pr, 3)
                        nd['predicted_spend'] = np.round(sp); nd['cluster'] = ck
                        m1, m2 = st.columns(2)
                        m1.metric("Interested", f"{pd_.sum()}/{len(pd_)}")
                        m2.metric("Avg Spend", f"{sp.mean():.0f} AED")
                        st.dataframe(nd.head(20), use_container_width=True)
                        st.download_button("⬇️ Download", nd.to_csv(index=False), "predictions.csv", "text/csv", use_container_width=True)
            except Exception as e: st.error(str(e))


# ================================================================
#  PAGE 6: DATA DICTIONARY
# ================================================================
elif page == "ℹ️ Data Dictionary":
    st.title("ℹ️ Data Dictionary")
    st.dataframe(df_dict, use_container_width=True, height=500)
    s1, s2 = st.columns(2)
    s1.metric("Raw Rows", f"{len(df_raw):,}"); s2.metric("Encoded Cols", f"{df_enc.shape[1]}")
    if '_persona' in df_raw.columns:
        pc = df_raw['_persona'].value_counts()
        fig = px.bar(x=pc.index, y=pc.values, color=pc.index, color_discrete_sequence=px.colors.qualitative.Set2, labels={'x': 'Persona', 'y': 'Count'}, title="Persona Distribution")
        fig.update_layout(height=350, showlegend=False); st.plotly_chart(fig, use_container_width=True)
