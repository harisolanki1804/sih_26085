# VARUNA — Idea Submission

**Problem Statement:** SIH26077 — AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting
**Organisation:** Ministry of Earth Sciences (MoES) · **Theme:** Disaster Management · **Category:** Software
**Team:** Team Varuna · Team ID 135235

**Portal field limits:** Idea Title 100 chars · Idea Description 50,000 chars · Abstract/Summary 10,000 chars · Idea Template PDF up to 10 MB

> The three fields below are sized to the portal limits and counted. Paste them as-is.

---

## Idea Title — 99 / 100 characters

VARUNA: AI-Driven Hyper-Local Severe Weather Nowcasting — Thunderstorms, Cloudbursts & Flash Floods

---

## Idea Description — 2,786 / 50,000 characters

VARUNA is an AI-driven early warning system for severe weather nowcasting. It predicts three hazards - severe thunderstorms, cloudbursts and flash floods - 2 to 6 hours in advance over a hyperlocal area, and then translates each prediction into street-level flood depth. Rather than issuing one district-level warning, it produces a multi-task output on a unified spatiotemporal grid: an independent probability for each hazard family, a hazard severity score, and a predicted inundation depth.

Separating the hazards is the core design decision: the system reports thunderstorm, cloudburst and flash-flood risk independently and flags which dominates, so a responder knows whether to prepare for lightning and squall, for a short-duration extreme downpour, or for accumulated street inundation. The same engine serves both the forecaster and the field operator.

The system combines satellite observations, atmospheric reanalysis, terrain and drainage data on that unified grid. Its Physics-Informed Hybrid Risk Engine uses a physical drainage and runoff flow baseline and AI-based residual learning to correct the errors in that baseline, instead of learning hydrology from scratch. Every prediction carries an explainable score using SHAP (SHapley Additive exPlanations) and a statistically calibrated confidence range using conformal prediction. VARUNA also provides an interactive Digital Twin that allows operators to rewind and replay historical events, and a What-If Simulator to modify inputs such as rainfall intensity or drainage conditions and observe the resulting risk across all three hazards.

A locally-run offline GenAI layer allows operators to ask questions in plain language. It retrieves and explains the system's structured outputs - the three hazard probabilities, SHAP breakdowns, trust metrics and historical analogs - without generating or inventing risk values. Inference, scoring, explanation and the GenAI narrator continue locally using the most recently cached data when connectivity is unavailable.

The system is designed using free/open datasets and locally-run models, with a prototype focused on a flood-prone pilot region.

Integrated technologies: AI/ML — multi-task learning across three hazard heads, spatiotemporal nowcasting with ConvLSTM and attention, physics-informed residual learning and predictive modelling; Explainable AI — SHAP-based explanations and conformal prediction for calibrated confidence ranges; Geospatial Technology — DEM processing, terrain analysis, drainage-network modelling and street-level flood estimation; Generative AI — a locally deployed offline LLM with RAG for operator-facing explanations; Data Fusion — INSAT-3D/3DR satellite observations, IMDAA atmospheric reanalysis, rainfall, terrain and drainage data.

---

## Abstract / Summary — 1,439 / 10,000 characters

VARUNA is an AI-driven hyperlocal early warning system and Digital Twin for severe weather and urban flash floods. From the same fused inputs it nowcasts three hazards 2-6 hours ahead - severe thunderstorms, cloudbursts and flash floods - and translates them into street-level flood depth using DEM and drainage-network modelling. The system fuses satellite observations, atmospheric reanalysis, rainfall, DEM and drainage data. A multi-modal spatiotemporal nowcasting network (ConvLSTM with attention) analyses dynamic atmospheric predictors, while a multi-task architecture produces hyperlocal probability maps for thunderstorms, cloudbursts and flash floods, each alongside a severity score and a depth estimate. VARUNA uses physics-informed residual learning, SHAP-based explainability, and conformal prediction to provide physically grounded predictions, contributing factors and statistically calibrated confidence ranges. Its Digital Twin supports historical replay and What-If simulation.

A fully offline, locally-run GenAI layer answers operator questions from cached system outputs and alert history, and explains why each hazard was flagged. The prototype also includes automated alerting, closed-loop validation against citizen reports, and a state/city selector with Mumbai as the live pilot and other states represented as roadmap views, demonstrated over a 90-cell urban grid and a 72-step replay of a full storm lifecycle.

---

## Technology Bucket — single-select dropdown

**Select:** Artificial Intelligence & Machine Learning

Use the combined entry **"AI/ML & Geospatial Technology"** if the dropdown offers one; otherwise the AI/ML bucket is the correct choice, since the problem statement itself is titled "AI-Driven". The geospatial, GenAI and data-fusion components are already named inside the Idea Description and Abstract above, so nothing is lost by the dropdown being narrow.

For reference, VARUNA integrates:

* **AI/ML:** Multi-task learning across three hazard heads (thunderstorm, cloudburst, flash flood), spatiotemporal nowcasting with ConvLSTM and attention, physics-informed residual learning and predictive modelling.
* **Explainable AI:** SHAP-based explanations and conformal prediction for calibrated confidence ranges.
* **Geospatial Technology:** DEM processing, terrain analysis, drainage-network modelling and street-level flood estimation.
* **Generative AI:** Locally deployed offline LLM with RAG for operator-facing explanations.
* **Data Fusion:** INSAT-3D/3DR satellite observations, IMDAA atmospheric reanalysis, rainfall, terrain and drainage data.

---

## Idea Template — separate upload

Upload the portal's **provided template** as a PDF, under 10 MB. Use the portal's *Download Template* link — this is not the same document as the SIH idea presentation deck.

---

## YouTube Link — optional

&lt;paste the 2:10 demo video URL here&gt;

---

## Before you submit

The portal warns that **no changes are entertained over email or call after submission**, so verify these on the saved draft:

* **Idea Title** kept all 99 characters — it is the only field with no room to spare, and some portals truncate silently when rendering.
* The em dash in the title saved correctly. If it displays as `?` or `â€"`, replace it with a plain hyphen; the field still fits at 99 characters.
* **Team ID** reads 135235.
* The problem-statement heading on the form matches **SIH26077 — AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting**.
* **Save as Draft** first, reopen it, then confirm the three text fields before final submission.
