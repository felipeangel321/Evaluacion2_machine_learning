"""Inteligencia Musical — aplicación demo de los modelos entrenados.

Ejecutar desde la raíz del proyecto (después de `kedro run`):

    streamlit run app/app.py

Usa los modelos finales que eligió el proyecto: Gradient Boosting optimizado
(clasificación y regresión) y K-Means k = 6 (segmentación). No reentrena nada.
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

ACCENT = "#1A7A4C"   # verde del proyecto (mismo de la presentación)
MUTED = "#8A8F98"

st.set_page_config(page_title="Inteligencia Musical", page_icon="🎵", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos entrenados…")
def get_artifacts():
    return core.load_artifacts()


@st.cache_data(show_spinner="Puntuando el catálogo de test…")
def get_test_predictions():
    art = get_artifacts()
    pred = core.predict(art["test"], art)
    return pd.concat([art["test"], pred], axis=1)


art = get_artifacts()
scored = get_test_predictions()
scored["hit_real"] = (scored["popularity"] >= core.HIT_THRESHOLD).astype(int)

# ── Barra lateral ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Inteligencia Musical")
    st.caption("EFT MLY1101 · Caso C Spotify")
    st.markdown(
        "Demo de los modelos entrenados con el pipeline Kedro. "
        "**No reentrena nada**: carga los `.joblib` de `models/trained_models/`."
    )
    st.divider()
    st.markdown("**Modelos en uso**")
    st.markdown(
        f"- Clasificación: Gradient Boosting · F1 0.680 · ROC-AUC 0.884\n"
        f"- Regresión: Gradient Boosting · MAE {core.MAE_TEST}\n"
        f"- Segmentación: K-Means k = 6"
    )
    st.divider()
    st.caption(
        f"Catálogo de test: {len(scored):,} canciones que el modelo nunca vio "
        "(split 80/20 estratificado, semilla 42).".replace(",", ".")
    )

tab_sim, tab_promo, tab_seg, tab_info = st.tabs(
    ["🎯 Simulador de canción", "📈 Qué promocionar", "🧩 Segmentos", "ℹ️ Cómo funciona"]
)

# ── 1. Simulador ───────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("¿Será un hit esta canción?")
    st.caption(
        "Elige una canción real del catálogo de test (el modelo no la vio al entrenar) "
        "o diseña una nueva. Mueve las variables para ver cómo cambia la predicción."
    )

    origen = st.radio("Partir de", ["Una canción del catálogo", "Una canción nueva"], horizontal=True)

    if origen == "Una canción del catálogo":
        labels = (scored["track_name"] + " — " + scored["artists"].str.replace(";", ", ")
                  + "  (" + scored["track_genre"] + ")")
        idx = st.selectbox(
            "Busca por nombre, artista o género", scored.index,
            format_func=lambda i: labels[i],
            index=int(scored["popularity"].idxmax()),
        )
        base = scored.loc[idx, art["features"].columns].copy()
        key_sfx = str(scored.loc[idx, "track_id"])
    else:
        feats = art["features"]
        base = feats.median(numeric_only=True)
        base["track_genre"] = "pop"
        base["key"] = int(feats["key"].mode()[0])
        base["time_signature"] = 4
        base["mode"] = 1
        base["explicit"] = 0
        base["n_artists"] = 1
        base["track_id"], base["artists"], base["album_name"], base["track_name"] = "nueva", "", "", "Canción nueva"
        base["popularity"] = float("nan")
        idx = None
        key_sfx = "nueva"

    col_in, col_out = st.columns([1.1, 1], gap="large")

    with col_in:
        genre = st.selectbox(
            "Género", art["genres"], index=art["genres"].index(base["track_genre"]),
            key=f"genre_{key_sfx}",
        )
        values = {}
        c1, c2 = st.columns(2)
        for i, (var, (lo, hi, step, help_txt)) in enumerate(core.AUDIO_SLIDERS.items()):
            with (c1 if i % 2 == 0 else c2):
                values[var] = st.slider(
                    var, lo, hi, float(min(max(base[var], lo), hi)), step,
                    help=help_txt, key=f"{var}_{key_sfx}",
                )
        with st.expander("Más variables (explícita, artistas, tonalidad)"):
            e1, e2, e3 = st.columns(3)
            explicit = e1.checkbox("Explícita", bool(base["explicit"]), key=f"exp_{key_sfx}")
            n_artists = e2.number_input("N.º de artistas", 1, 20, int(base["n_artists"]), key=f"na_{key_sfx}")
            mode = e3.radio("Modo", [1, 0], index=0 if int(base["mode"]) == 1 else 1,
                            format_func=lambda m: "Mayor" if m else "Menor", key=f"mode_{key_sfx}")

    song = base.to_frame().T.copy()
    for var, v in values.items():
        song[var] = v
    song["track_genre"] = genre
    song["explicit"] = int(explicit)
    song["n_artists"] = n_artists
    song["is_collab"] = int(n_artists > 1)
    song["mode"] = mode
    for c in ["key", "time_signature"]:
        song[c] = int(base[c])
    num_cols = [c for c in song.columns if c not in ("track_id", "artists", "album_name", "track_name", "track_genre")]
    song[num_cols] = song[num_cols].astype(float)
    # Las categóricas deben quedar enteras: get_dummies sobre 4.0 crearía "time_signature_4.0"
    for c in ["key", "time_signature", "mode", "explicit", "n_artists", "is_collab"]:
        song[c] = song[c].astype(int)

    pred = core.predict(song.reset_index(drop=True), art).iloc[0]
    original = scored.loc[idx] if idx is not None else None

    with col_out:
        prob = pred["prob_hit"]
        decision = "✅ Promocionar" if prob >= 0.5 else "⏸️ No priorizar"
        delta_prob = None
        delta_pop = None
        if original is not None:
            if abs(prob - original["prob_hit"]) > 1e-9:
                delta_prob = f"{(prob - original['prob_hit']) * 100:+.1f} pts vs. original"
            if abs(pred["pop_estimada"] - original["pop_estimada"]) > 1e-9:
                delta_pop = f"{pred['pop_estimada'] - original['pop_estimada']:+.1f} vs. original"

        m1, m2 = st.columns(2)
        m1.metric("Probabilidad de hit", f"{prob:.0%}", delta_prob, border=True)
        m2.metric("Popularidad estimada", f"{pred['pop_estimada']:.0f} / 100", delta_pop,
                  help=f"Error medio del modelo en test: ±{core.MAE_TEST} puntos", border=True)
        st.markdown(f"### {decision}")
        st.caption(f"Umbral 0.5 · hit = popularidad ≥ {core.HIT_THRESHOLD} · "
                   f"rango probable de popularidad: {max(1, pred['pop_estimada'] - core.MAE_TEST):.0f}–"
                   f"{min(100, pred['pop_estimada'] + core.MAE_TEST):.0f}")

        seg_name, seg_rec = core.SEGMENTS[int(pred["segmento"])]
        st.info(f"**Segmento de sonido: {seg_name}**  \n{seg_rec}")

        if original is not None:
            real_hit = "sí fue hit" if original["popularity"] >= core.HIT_THRESHOLD else "no fue hit"
            st.markdown(
                f"**Dato real:** popularidad **{original['popularity']:.0f}** ({real_hit}). "
                f"Predicción original del modelo: {original['prob_hit']:.0%} de hit, "
                f"{original['pop_estimada']:.0f} de popularidad."
            )

        if genre in core.BIASED_GENRES:
            st.warning(
                "⚠️ **Sesgo medido:** en música latina urbana el error es 2.3× mayor "
                "(MAE 19.7) y el modelo subestima la popularidad. Esta predicción "
                "requiere revisión humana."
            )
        st.caption("💡 Prueba a cambiar solo el **género**: es la variable que más pesa en el modelo "
                   "(importancia por permutación 0.32 en ROC-AUC, contra ≤ 0.013 de cualquier variable de audio).")

# ── 2. Qué promocionar ─────────────────────────────────────────────────────
with tab_promo:
    st.subheader("¿Qué canciones promocionar con un presupuesto limitado?")
    st.caption(
        "El clasificador ordena el catálogo de test por probabilidad de hit. "
        "Eliges qué porcentaje alcanza a promocionar y se compara con elegir al azar."
    )

    f1c, f2c = st.columns([1, 2])
    budget = f1c.slider("Presupuesto: % del catálogo a promocionar", 1, 30, 10)
    genres_sel = f2c.multiselect("Filtrar géneros (opcional, p. ej. un sello de un nicho)", art["genres"])

    pool = scored if not genres_sel else scored[scored["track_genre"].isin(genres_sel)]
    if len(pool) == 0:
        st.stop()
    ranked = pool.sort_values("prob_hit", ascending=False)
    k = max(1, round(len(ranked) * budget / 100))
    top = ranked.head(k)

    precision = top["hit_real"].mean()
    base_rate = pool["hit_real"].mean()
    recall = top["hit_real"].sum() / max(1, pool["hit_real"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Canciones promocionadas", f"{k:,}".replace(",", "."), border=True)
    k2.metric("Son hits reales", f"{precision:.0%}", f"{(precision - base_rate) * 100:+.0f} pts vs. azar", border=True)
    k3.metric("Veces mejor que el azar", f"{precision / base_rate:.1f}×" if base_rate > 0 else "—", border=True)
    k4.metric("Hits del catálogo cubiertos", f"{recall:.0%}", border=True)
    st.caption(f"Elegir al azar daría {base_rate:.1%} de hits en este catálogo.")

    curve = core.promotion_curve(pool[["prob_hit"]], pool["hit_real"], budgets=range(1, 51))
    line = alt.Chart(curve).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=alt.X("presupuesto_%:Q", title="% del catálogo promocionado"),
        y=alt.Y("% hits en lo promocionado:Q", title="% de hits reales", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("presupuesto_%:Q", title="Presupuesto (%)"),
                 alt.Tooltip("% hits en lo promocionado:Q", title="% hits", format=".1f")],
    )
    hover = alt.selection_point(on="pointerover", nearest=True, empty=False, fields=["presupuesto_%"])
    points = line.mark_point(color=ACCENT, size=80, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).add_params(hover)
    rule = alt.Chart(pd.DataFrame({"y": [base_rate * 100]})).mark_rule(
        color=MUTED, strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q")
    rule_lbl = alt.Chart(pd.DataFrame({"y": [base_rate * 100], "x": [50], "t": ["al azar"]})).mark_text(
        align="right", dy=-8, color=MUTED, fontSize=13).encode(x="x:Q", y="y:Q", text="t:N")
    sel = alt.Chart(pd.DataFrame({"x": [budget]})).mark_rule(color=ACCENT, opacity=0.35).encode(x="x:Q")
    st.altair_chart((rule + rule_lbl + sel + line + points).properties(height=300), width="stretch")

    st.markdown(f"**Lista de promoción** (primeras {min(k, 200)} de {k})")
    show = top.head(200).assign(
        artistas=lambda d: d["artists"].str.replace(";", ", "),
        prob=lambda d: (d["prob_hit"] * 100).round(1),
        pop_est=lambda d: d["pop_estimada"].round(0),
        hit=lambda d: d["hit_real"].map({1: "✅", 0: "—"}),
    )[["track_name", "artistas", "track_genre", "prob", "pop_est", "popularity", "hit"]]
    show.columns = ["Canción", "Artistas", "Género", "Prob. hit (%)", "Pop. estimada", "Pop. real", "¿Fue hit?"]
    st.dataframe(show, hide_index=True, width="stretch", height=360)
    st.download_button(
        "⬇️ Descargar lista completa (CSV)",
        top[["track_id", "track_name", "artists", "track_genre", "prob_hit", "pop_estimada", "popularity"]]
        .to_csv(index=False).encode("utf-8"),
        file_name=f"lista_promocion_top{budget}.csv", mime="text/csv",
    )

# ── 3. Segmentos ───────────────────────────────────────────────────────────
with tab_seg:
    st.subheader("Seis segmentos de sonido, con distinto potencial")
    st.caption("K-Means sobre 10 variables de audio, **sin usar género ni popularidad**. "
               "Aun así, la tasa de hits va de 4% a 30% según el segmento.")

    prof = pd.read_csv(core.ROOT / "results/metrics/cluster_profiles.csv")
    prof["Segmento"] = prof["cluster"].map(lambda c: core.SEGMENTS[c][0])
    prof["Recomendación"] = prof["cluster"].map(lambda c: core.SEGMENTS[c][1])
    prof = prof.sort_values("tasa_hit", ascending=False)

    bars = alt.Chart(prof).mark_bar(color=ACCENT, cornerRadiusEnd=4, size=26).encode(
        y=alt.Y("Segmento:N", sort="-x", title=None),
        x=alt.X("tasa_hit:Q", title="% de hits", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("Segmento:N"), alt.Tooltip("tasa_hit:Q", title="% hits", format=".0%"),
                 alt.Tooltip("pct_canciones:Q", title="% del catálogo", format=".0%"),
                 alt.Tooltip("popularity_media:Q", title="Pop. media", format=".1f")],
    )
    labels_txt = bars.mark_text(align="left", dx=6, color="#16181D", fontSize=13).encode(
        text=alt.Text("tasa_hit:Q", format=".0%"))
    st.altair_chart((bars + labels_txt).properties(height=280), width="stretch")

    tbl = prof[["Segmento", "pct_canciones", "popularity_media", "tasa_hit", "top_5_generos", "Recomendación"]].copy()
    tbl["pct_canciones"] = (tbl["pct_canciones"] * 100).round(0).astype(int).astype(str) + "%"
    tbl["tasa_hit"] = (tbl["tasa_hit"] * 100).round(0).astype(int).astype(str) + "%"
    tbl.columns = ["Segmento", "% catálogo", "Pop. media", "% hits", "Géneros típicos", "Recomendación"]
    st.dataframe(tbl, hide_index=True, width="stretch")

    st.markdown("**Perfil de sonido de un segmento** (promedio de cada variable, escala 0–1)")
    seg_pick = st.selectbox("Segmento", prof["Segmento"].tolist())
    row = prof[prof["Segmento"] == seg_pick].iloc[0]
    feats01 = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]
    overall = art["features"][feats01].mean()
    prof_df = pd.DataFrame({"variable": feats01, "Segmento": [row[f] for f in feats01],
                            "Promedio catálogo": overall.values})
    seg_bars = alt.Chart(prof_df).mark_bar(color=ACCENT, cornerRadiusEnd=4, size=18).encode(
        y=alt.Y("variable:N", sort=feats01, title=None),
        x=alt.X("Segmento:Q", title="valor medio", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("Segmento:Q", title=seg_pick, format=".2f"),
                 alt.Tooltip("Promedio catálogo:Q", format=".2f")],
    )
    avg_tick = alt.Chart(prof_df).mark_tick(color="#16181D", thickness=2, size=26).encode(
        y=alt.Y("variable:N", sort=feats01), x="Promedio catálogo:Q")
    st.altair_chart((seg_bars + avg_tick).properties(height=260), width="stretch")
    st.caption("Barra verde = segmento elegido · marca negra = promedio de todo el catálogo.")

# ── 4. Cómo funciona ───────────────────────────────────────────────────────
with tab_info:
    st.subheader("De los datos a la decisión")
    st.markdown(
        """
```
Spotify_Tracks_Dataset.csv ──► kedro run (35 nodos, CRISP-DM) ──► models/*.joblib ──► esta app
   114.000 canciones            limpieza · features · tuning          modelos finales       predice,
                                ensambles · evaluación                + scaler               ordena y segmenta
```
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Qué hace la app con cada canción**")
        st.markdown(
            "1. Crea las mismas variables que el pipeline (duración en minutos, n.º de artistas, colaboración).\n"
            "2. Aplica One-Hot a género, tonalidad y compás → 144 variables.\n"
            "3. Escala con el **mismo StandardScaler** ajustado en train (`data/06_models/scaler.pkl`).\n"
            "4. Gradient Boosting predice la probabilidad de hit y la popularidad; K-Means asigna el segmento."
        )
        st.markdown("**Por qué Gradient Boosting**")
        st.markdown(
            "Gana en clasificación (F1 0.680) y empata con Stacking en regresión (8.65 vs. 8.63), "
            "siendo 5 a 11 veces más liviano y más fácil de explicar."
        )
    with c2:
        st.markdown("**Límites y uso responsable**")
        st.markdown(
            "- Es una herramienta de **apoyo** al criterio editorial, no de reemplazo.\n"
            "- Subestima los éxitos masivos (> 60): dependen del artista y del marketing, que no están en los datos.\n"
            "- Sesgo medido en música latina urbana (error 2.3×): revisión humana obligatoria.\n"
            "- Promocionar solo lo predicho refuerza lo ya popular: mantener cuotas para nicho y artistas nuevos.\n"
            "- Sin datos de usuarios; el modelo no usa nombres de artistas. Datos de la API de Spotify: uso académico."
        )
    st.caption(f"Verificación: sobre el catálogo de test la app reproduce las métricas del informe "
               f"(F1 0.680 · ROC-AUC 0.884 · MAE {core.MAE_TEST}).")
