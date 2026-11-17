// Phase 2 evaluation deck: Heart Disease Risk Modelling.
// Numbers are read from the saved Phase 2 run (results/phase2/main/metrics) and
// the Phase 1 summary, passed in as data.json by prepare_data.py.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");
const D = require("./data.json");

const THEME = {
  name: "Heart Disease Phase 2",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2130", lt1: "FFFFFF", dk2: "141A26", lt2: "F2F3F6",
    accent1: "0173B2", accent2: "DE8F05", accent3: "029E73", accent4: "A3172F",
    accent5: "5B6475", accent6: "D5D9E0", hlink: "0173B2", folHlink: "5B6475",
  },
};
const HEX = THEME.colors;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in, same canvas as the Phase 1 deck
  pres.title = "Heart Disease Risk Modelling - Phase 2";
  pres.author = "Hardik, Ayush, Akshith, Vansh, Kabir";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const FOOT = "Heart disease risk modelling · Phase 2 · academic study, not a diagnostic tool";
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    objects: [
      { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.75, fontFace: "Cambria", fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "DARK", background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.5, h: 1.2, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });

  const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.08, objectName: name });
  const txt = (s, t, o) => s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontFace: "Calibri", color: C.text1, valign: "top" }, o));
  const numDot = (s, n, x, y, fill = C.accent4) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill }, line: { type: "none" } });
    txt(s, String(n), { x, y, w: 0.42, h: 0.42, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle" });
  };
  const who = (s, name, dark = false) => txt(s, "Presenter: " + name, { x: 8.6, y: 7.05, w: 3.6, h: 0.3, fontSize: 10, bold: true, align: "right", color: dark ? HEX.accent6 : C.accent4 });
  const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };

  const heart = await icon(fa.FaHeartbeat, HEX.accent4);
  const iconCheck = await icon(fa.FaCheckCircle, "FFFFFF");
  const iconHospital = await icon(fa.FaHospital, "FFFFFF");
  const iconBalance = await icon(fa.FaBalanceScale, "FFFFFF");
  const iconWarn = await icon(fa.FaExclamationTriangle, "FFFFFF");

  // ---------- 1. Title ----------
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addImage({ data: heart, x: 0.8, y: 0.8, w: 0.75, h: 0.75 });
  s.addText("Heart Disease Risk Modelling", { placeholder: "title" });
  txt(s, "Phase 2: four hospital cohorts, a regularised neural network and a Bayesian network–MLP stack, tuned by TPE without hand-written loops",
    { x: 0.8, y: 3.25, w: 10.5, h: 1.0, fontSize: 20, color: HEX.accent6 });
  txt(s, "Ayush  ·  Vansh  ·  Hardik  ·  Akshith  ·  Kabir", { x: 0.8, y: 5.1, w: 10, h: 0.45, fontSize: 18, bold: true, color: C.background1 });
  txt(s, "Advanced Machine Learning  ·  Newton School of Technology, Rishihood University  ·  Supervisor: Mr. Shobhit Johri  ·  November 2026",
    { x: 0.8, y: 5.6, w: 11.5, h: 0.4, fontSize: 14, color: HEX.accent6 });
  who(s, "Kabir", true);
  s.addNotes("Presenter: Kabir. Introduce the team and the plan of the talk. Phase 2 extends Phase 1; it does not replace it. We will spend two slides on Phase 1 and the feedback we received, then show how each piece of feedback was addressed and what the results say.");

  // ---------- 2. Phase 1 recap ----------
  pres.addSection({ title: "Phase 1" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Phase 1" });
  s.addText("Phase 1 recap: six classical models on Cleveland", { placeholder: "title" });
  card(s, 0.6, 1.4, 5.0, 5.35);
  txt(s, "What we built", { x: 0.9, y: 1.6, w: 4.4, h: 0.4, fontSize: 18, bold: true, fontFace: "Cambria" });
  txt(s, [
    { text: "303 Cleveland patients, 13 predictors, disease = num 1–4", options: { bullet: true, breakLine: true } },
    { text: "Bayesian network with an expert DAG, Naive Bayes, Logistic Regression, SVM, Random Forest, Gradient Boosting", options: { bullet: true, breakLine: true } },
    { text: "5×5 repeated nested CV, preprocessing fitted inside training folds", options: { bullet: true, breakLine: true } },
    { text: "Tuning: a grid expanded and looped over by our own code", options: { bullet: true, breakLine: true } },
    { text: "Streamlit app and project report", options: { bullet: true } },
  ], { x: 0.9, y: 2.1, w: 4.45, h: 3.4, fontSize: 15, paraSpaceAfter: 6 });
  txt(s, [{ text: "Result: ", options: { bold: true, color: C.accent4 } },
    { text: "all six models were good (ROC AUC 0.890–0.910) and five were statistically tied." }],
    { x: 0.9, y: 5.6, w: 4.45, h: 0.9, fontSize: 15 });
  s.addChart(pres.charts.BAR, [{ name: "Pooled ROC AUC", labels: D.p1.labels, values: D.p1.auc }], Object.assign({
    x: 6.0, y: 1.4, w: 6.75, h: 5.35, barDir: "bar", chartColors: [HEX.accent5],
    showTitle: true, title: "Phase 1 pooled out-of-fold ROC AUC (303 patients)", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: 0.85, valAxisMaxVal: 0.92, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valGridLine: { color: HEX.accent6, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
  }, chartText));
  who(s, "Ayush");
  s.addNotes("Presenter: Ayush. Phase 1 compared six classical models on the Cleveland cohort only, under 5x5 repeated nested cross-validation. All discriminated well, AUC 0.890 to 0.910, and only Gradient Boosting was slightly behind. Two weaknesses mattered for Phase 2: one centre only, and a hand-written grid loop for tuning.");

  // ---------- 3. Feedback -> what changed ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Phase 1" });
  s.addText("Phase 1 feedback and how Phase 2 answers it", { placeholder: "title" });
  const fb = [
    ["Use all available heart disease datasets", "All four independent UCI cohorts: 920 patients, cohort provenance, hashes, missingness audit", "Ayush"],
    ["No manual loops for hyperparameter selection", "Hand-written grid loop removed; OptunaSearchCV owns every candidate and inner CV", "Akshith"],
    ["Integrate ML with deep learning, advanced HPO", "Regularised PyTorch MLP + cross-fitted BN–MLP stack; TPE (Bayesian optimisation)", "Hardik, Vansh"],
    ["IEEE-format report", "IEEEtran paper, every table generated from one saved run and audited", "All five"],
  ];
  txt(s, "Feedback", { x: 1.25, y: 1.35, w: 3.6, h: 0.35, fontSize: 13, bold: true, color: C.accent5 });
  txt(s, "What we did", { x: 5.15, y: 1.35, w: 5.4, h: 0.35, fontSize: 13, bold: true, color: C.accent5 });
  txt(s, "Owner", { x: 10.85, y: 1.35, w: 1.8, h: 0.35, fontSize: 13, bold: true, color: C.accent5 });
  fb.forEach((r, i) => {
    const y = 1.8 + i * 1.22;
    card(s, 0.6, y, 12.1, 1.06);
    numDot(s, i + 1, 0.75, y + 0.32);
    txt(s, r[0], { x: 1.25, y: y + 0.1, w: 3.65, h: 0.86, fontSize: 16, bold: true, fontFace: "Cambria", valign: "middle" });
    txt(s, r[1], { x: 5.15, y: y + 0.1, w: 5.45, h: 0.86, fontSize: 15, valign: "middle" });
    txt(s, r[2], { x: 10.85, y: y + 0.1, w: 1.75, h: 0.86, fontSize: 15, bold: true, color: C.accent4, valign: "middle" });
  });
  txt(s, "Plus, from the Phase 2 brief: architecture logic, regularisation, ablations, cross-cohort validation, explainability and a live demo (Kabir).",
    { x: 0.6, y: 6.7, w: 12.1, h: 0.3, fontSize: 13, italic: true, color: C.accent5 });
  who(s, "Ayush");
  s.addNotes("Presenter: Ayush. Each row is one point of the Phase 1 feedback and the person who owns the answer. Integration is our design choice: the feedback asked for ML and DL to be integrated, and we chose a leakage-safe stack of the Bayesian network and the MLP because they make complementary assumptions.");

  // ---------- 4. Data ----------
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("920 patients from four hospitals, very unevenly measured", { placeholder: "title" });
  const stats = [["920", "patients"], ["4", "cohorts"], ["55%", "have disease"], ["66%", "missing ca"]];
  stats.forEach((st, i) => {
    txt(s, st[0], { x: 0.6 + i * 1.55, y: 1.4, w: 1.5, h: 0.75, fontSize: 36, bold: true, color: C.accent4, fontFace: "Cambria" });
    txt(s, st[1], { x: 0.6 + i * 1.55, y: 2.15, w: 1.5, h: 0.35, fontSize: 13, color: C.accent5 });
  });
  card(s, 0.6, 2.75, 6.0, 4.0);
  txt(s, [
    { text: "Cleveland 303 · Hungarian 294 · Switzerland 123 · Long Beach VA 200", options: { bullet: true, breakLine: true } },
    { text: "Only the four processed files: raw and reprocessed versions are the same patients and would double-count", options: { bullet: true, breakLine: true } },
    { text: "Missing outside Cleveland: ca 611, thal 486, slope 309 of 920", options: { bullet: true, breakLine: true } },
    { text: "Imputation, scaling and bins fitted inside each training fold; no complete-case deletion", options: { bullet: true, breakLine: true } },
    { text: "Cohort kept for splitting and reporting, never used as a predictor", options: { bullet: true } },
  ], { x: 0.85, y: 2.95, w: 5.55, h: 3.7, fontSize: 15, paraSpaceAfter: 6 });
  s.addChart(pres.charts.BAR, [{ name: "Disease prevalence", labels: D.cohort.labels, values: D.cohort.prev }], Object.assign({
    x: 7.0, y: 1.4, w: 5.7, h: 5.35, barDir: "col", chartColors: [HEX.accent1],
    showTitle: true, title: "Disease prevalence by cohort", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", dataLabelFontSize: 13, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valGridLine: { color: HEX.accent6, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  }, chartText));
  txt(s, "Switzerland has only 8 disease-free patients, so its specificity is not interpretable.",
    { x: 7.0, y: 6.75, w: 5.7, h: 0.3, fontSize: 11, italic: true, color: C.accent5 });
  who(s, "Ayush");
  s.addNotes("Presenter: Ayush. The extra cohorts bring patients but also shift: prevalence ranges from 36 percent in Hungary to 94 percent in Switzerland, and the two strongest Phase 1 predictors, ca and thal, are mostly missing outside Cleveland. Because missingness is decided by the hospital, it is not random. We impute inside training folds only. For the Bayesian network we chose fold-wise mode imputation rather than a 'missing' state, because that state would effectively encode the hospital.");

  // ---------- 5. Models: MLP and stack ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Deep-learning component and its integration with the BN", { placeholder: "title" });
  txt(s, "Regularised MLP (PyTorch, sklearn-compatible)", { x: 0.6, y: 1.35, w: 6.0, h: 0.4, fontSize: 18, bold: true, fontFace: "Cambria" });
  const layers = [["13 raw\ninputs", C.accent6], ["impute,\nscale,\none-hot\nd = 20", C.accent6], ["32\nReLU\ndropout", C.accent4], ["16\nReLU\ndropout", C.accent4], ["1\nsigmoid", C.accent2]];
  layers.forEach((l, i) => {
    const x = 0.6 + i * 1.22;
    const dark = i >= 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.95, w: 1.02, h: 1.35, fill: { color: l[1] }, line: { type: "none" }, rectRadius: 0.08 });
    txt(s, l[0], { x, y: 1.95, w: 1.02, h: 1.35, fontSize: 13, align: "center", valign: "middle", color: dark ? C.background1 : C.text1, bold: dark });
    if (i < layers.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.04, y: 2.52, w: 0.16, h: 0.22, fill: { color: C.accent5 }, line: { type: "none" } });
  });
  txt(s, [
    { text: "BCE loss, Adam, weight decay, dropout, batch 16/32", options: { bullet: true, breakLine: true } },
    { text: "Early stopping on a 20% split of the raw training rows; preprocessing fitted on the other 80% only", options: { bullet: true, breakLine: true } },
    { text: "Layout searched: 16, 16-8 or 32-16 (353 to 1,217 weights); 32-16 chosen in 4 of 5 folds", options: { bullet: true, breakLine: true } },
    { text: "No CNN or LSTM: patient features have no spatial or time order", options: { bullet: true } },
  ], { x: 0.6, y: 3.55, w: 5.95, h: 3.2, fontSize: 15, paraSpaceAfter: 6 });

  card(s, 7.0, 1.35, 5.7, 5.4, C.text2);
  txt(s, "Cross-fitted BN–MLP stack", { x: 7.3, y: 1.55, w: 5.1, h: 0.4, fontSize: 18, bold: true, fontFace: "Cambria", color: C.background1 });
  const steps = [
    "Split the training rows into 3 folds",
    "Fit BN and MLP on 2 folds, predict the 3rd: out-of-fold P(disease) for every patient",
    "Fit a regularised logistic combiner on [p_BN, p_MLP]",
    "Refit BN and MLP on all training rows for prediction",
  ];
  steps.forEach((t, i) => {
    numDot(s, i + 1, 7.3, 2.2 + i * 0.95, C.accent4);
    txt(s, t, { x: 7.9, y: 2.15 + i * 0.95, w: 4.55, h: 0.85, fontSize: 15, color: C.background1, valign: "middle" });
  });
  txt(s, "The combiner never sees a probability from a model that was trained on that patient.",
    { x: 7.3, y: 6.0, w: 5.1, h: 0.6, fontSize: 13, italic: true, color: HEX.accent6 });
  who(s, "Hardik (MLP) \u00b7 Vansh (stack)");
  s.addNotes("Presenters: Hardik (MLP), then Vansh (stack). The MLP is small on purpose: 920 patients and 13 predictors. Early stopping is leakage-safe because the stopping rows are split off before the preprocessor is fitted. The stack follows stacked generalisation: the meta-learner must learn from out-of-fold predictions, otherwise it would learn to trust over-fitted in-sample probabilities. The BN brings expert structure and transfer; the MLP brings smooth nonlinear interactions.");

  // ---------- 6. HPO without loops ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Hyperparameters: TPE search, no hand-written loops", { placeholder: "title" });
  card(s, 0.6, 1.4, 5.6, 2.45);
  txt(s, "Phase 1 (removed)", { x: 0.85, y: 1.55, w: 5.1, h: 0.35, fontSize: 15, bold: true, color: C.accent4 });
  txt(s, "for params in expand_grid(grid):\n    score = inner_cv(model.set_params(**params))", { x: 0.85, y: 2.0, w: 5.2, h: 0.8, fontSize: 13, fontFace: "Courier New" });
  txt(s, "Our code iterated over every combination; cost grows exponentially with the number of hyperparameters.", { x: 0.85, y: 2.9, w: 5.1, h: 0.8, fontSize: 14 });
  card(s, 0.6, 4.1, 5.6, 2.65);
  txt(s, "Phase 2", { x: 0.85, y: 4.25, w: 5.1, h: 0.35, fontSize: 15, bold: true, color: C.accent3 });
  txt(s, "OptunaSearchCV(model, space,\n  sampler=TPESampler(seed), cv=3-fold,\n  scoring='neg_log_loss', n_trials=12,\n  refit=True)", { x: 0.85, y: 4.7, w: 5.2, h: 1.1, fontSize: 13, fontFace: "Courier New" });
  txt(s, "The library proposes, cross-validates and refits. TPE models good vs bad trials and samples where l(x)/g(x) is highest.", { x: 0.85, y: 5.85, w: 5.1, h: 0.8, fontSize: 14 });

  const ev = [["864", "TPE trials saved (480 pooled + 384 cross-cohort)"], ["8", "models searched by the same builder"], ["15–29", "BN smoothing now chosen inside the range (Phase 1: stuck at the top value in 25/25 folds)"]];
  ev.forEach((e, i) => {
    const y = 1.4 + i * 1.8;
    card(s, 6.6, y, 6.1, 1.6);
    txt(s, e[0], { x: 6.85, y: y + 0.25, w: 1.9, h: 1.1, fontSize: 36, bold: true, color: C.accent4, fontFace: "Cambria", valign: "middle" });
    txt(s, e[1], { x: 8.8, y: y + 0.2, w: 3.7, h: 1.2, fontSize: 15, valign: "middle" });
  });
  who(s, "Akshith");
  s.addNotes("Presenter: Akshith. 'No loops' means no team-written loop over parameter combinations; the library necessarily evaluates candidates internally. We still loop over outer folds, but that is evaluation, not selection. Log loss is the selection metric because it rewards calibrated probabilities. The best inner score is not the final performance: only the outer folds give that.");

  // ---------- 7. Evaluation design ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Two validation protocols, one leakage rule", { placeholder: "title" });
  const lv = [
    ["Outer split", "Pooled: 5 stratified folds of 920 patients.\nCross-cohort: train on 3 hospitals, test on the 4th.", C.text2, C.background1],
    ["TPE search", "OptunaSearchCV on outer-training rows only: 3-fold inner CV, 12 trials, refit best.", C.accent5, C.background1],
    ["Inside every fit", "Imputation and scaling, MLP early stopping, stack cross-fitting: all on training rows.", C.accent6, C.text1],
  ];
  lv.forEach((l, i) => {
    const x = 0.6, y = 1.45 + i * 1.3, w = 7.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.05, fill: { color: l[2] }, line: { type: "none" }, rectRadius: 0.08 });
    txt(s, l[0], { x: x + 0.25, y: y + 0.1, w: 2.0, h: 0.85, fontSize: 16, bold: true, color: l[3], fontFace: "Cambria", valign: "middle" });
    txt(s, l[1], { x: x + 2.3, y: y + 0.08, w: w - 2.5, h: 0.9, fontSize: 14, color: l[3], valign: "middle" });
    if (i < lv.length - 1) s.addShape(pres.shapes.DOWN_ARROW, { x: x + 3.35, y: y + 1.08, w: 0.3, h: 0.2, fill: { color: C.accent5 }, line: { type: "none" } });
  });
  txt(s, [
    { text: "Outputs: ", options: { bold: true } },
    { text: "out-of-fold probabilities for every patient → ROC AUC, PR AUC, Brier, log loss, calibration error, sensitivity and specificity at 0.5, per cohort, with patient-level bootstrap intervals and paired model comparisons." },
  ], { x: 0.6, y: 5.4, w: 7.0, h: 1.3, fontSize: 15 });
  card(s, 8.0, 1.45, 4.7, 5.3);
  s.addShape(pres.shapes.OVAL, { x: 8.25, y: 1.65, w: 0.6, h: 0.6, fill: { color: C.accent4 }, line: { type: "none" } });
  s.addImage({ data: iconHospital, x: 8.37, y: 1.77, w: 0.36, h: 0.36 });
  txt(s, "Why hold out a hospital?", { x: 9.0, y: 1.7, w: 3.5, h: 0.5, fontSize: 17, bold: true, fontFace: "Cambria", valign: "middle" });
  txt(s, [
    { text: "Random folds mix patients from the same hospital into training and test", options: { bullet: true, breakLine: true } },
    { text: "A new site has its own prevalence, tests and missing data", options: { bullet: true, breakLine: true } },
    { text: "Leave-one-cohort-out measures that shift directly; it is still historical data, not clinical validation", options: { bullet: true } },
  ], { x: 8.3, y: 2.5, w: 4.15, h: 4.0, fontSize: 15, paraSpaceAfter: 8 });
  who(s, "Vansh");
  s.addNotes("Presenter: Vansh. Every model sees identical outer folds. Outer test patients never influence preprocessing, hyperparameters, early stopping, the stack or the 0.5 threshold. The cross-cohort protocol is harder than random folds because the test hospital's distribution is absent from training.");

  // ---------- 8. Pooled results ----------
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Pooled results: MLP and stack tie the best classical models", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "ROC AUC", labels: D.labels, values: D.pooled.auc }], Object.assign({
    x: 0.6, y: 1.35, w: 6.6, h: 5.4, barDir: "bar", chartColors: D.colors,
    showTitle: true, title: "Pooled nested CV ROC AUC, 920 patients", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: 0.85, valAxisMaxVal: 0.90, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valGridLine: { color: HEX.accent6, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
  }, chartText));
  const rows = [[{ text: "Model", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } },
    { text: "Brier", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } },
    { text: "Log loss", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } },
    { text: "ECE", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }]];
  D.pooled.table.forEach((r) => rows.push(r.map((v, j) => ({ text: v, options: { bold: j === 0 || r[0].startsWith("MLP") || r[0].startsWith("BN–MLP"), fill: { color: (r[0] === "MLP" || r[0] === "BN–MLP stack") ? HEX.lt2 : HEX.lt1 } } }))));
  s.addTable(rows, { x: 7.6, y: 1.4, w: 5.1, colW: [2.0, 1.0, 1.1, 1.0], fontSize: 13, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: HEX.accent6 }, rowH: 0.34, align: "left", valign: "middle" });
  txt(s, [
    { text: "Paired bootstrap: ", options: { bold: true, color: C.accent4 } },
    { text: "MLP vs LR −0.001 and vs SVM −0.007 (both include 0). MLP and stack beat the BN by +0.013 and +0.014. The MLP is the best calibrated (ECE 0.018)." },
  ], { x: 7.6, y: 4.75, w: 5.1, h: 1.9, fontSize: 15 });
  who(s, "Hardik");
  s.addNotes("Presenter: Hardik. With random patient folds, all eight models sit between 0.875 and 0.895. The neural network adds no discrimination over logistic regression or the SVM; the paired intervals include zero. The stack is as good as its best constituent but not better. Calibration is good for all models; the MLP has the lowest expected calibration error.");

  // ---------- 9. Cross-cohort ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("A new hospital costs more AUC than the choice of model", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Pooled nested CV", labels: D.labels, values: D.pooled.auc },
    { name: "Mean AUC on held-out hospital", labels: D.labels, values: D.loco.mean },
  ], Object.assign({
    x: 0.6, y: 1.35, w: 7.4, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent6, HEX.accent4],
    showTitle: true, title: "ROC AUC: random patient folds vs leave-one-hospital-out", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: 0.70, valAxisMaxVal: 0.92, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 10, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valGridLine: { color: HEX.accent6, size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  }, chartText));
  const callouts = [
    ["−0.05 to −0.08", "AUC lost by every model when its test hospital is unseen, versus a 0.02 spread between models"],
    ["NB 0.81 · BN 0.80", "the simple probabilistic models transfer best; MLP worst (0.77)"],
    ["+0.017", "stack over MLP alone under hospital shift (95% CI +0.005 to +0.030): the BN stabilises the MLP"],
  ];
  callouts.forEach((c, i) => {
    const y = 1.35 + i * 1.82;
    card(s, 8.35, y, 4.35, 1.62);
    txt(s, c[0], { x: 8.6, y: y + 0.15, w: 3.9, h: 0.55, fontSize: 24, bold: true, color: C.accent4, fontFace: "Cambria" });
    txt(s, c[1], { x: 8.6, y: y + 0.72, w: 3.9, h: 0.85, fontSize: 14 });
  });
  who(s, "Vansh");
  s.addNotes("Presenter: Vansh. This is the most important result. Holding out a whole hospital lowers AUC by 0.046 to 0.079, more than the entire difference between models. The ranking changes: Naive Bayes and the Bayesian network transfer best, the MLP and SVM worst. Here the integration earns its place: the stack is significantly better than the MLP alone. Cleveland suffers most when held out, because it is the only cohort that measured ca and thal. Switzerland's numbers rest on eight healthy patients.");

  // ---------- 10. Ablations ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Ablations: at 920 patients the regularisers barely matter", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "ROC AUC", labels: D.abl.labels, values: D.abl.auc }], Object.assign({
    x: 0.6, y: 1.35, w: 7.4, h: 5.4, barDir: "bar", chartColors: D.abl.colors,
    showTitle: true, title: "5-fold CV ROC AUC, one change at a time (fold SD 0.014–0.027)", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: 0.86, valAxisMaxVal: 0.90, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valGridLine: { color: HEX.accent6, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
  }, chartText));
  const ab = [
    ["MLP variants (red)", "Removing dropout, weight decay or early stopping, or using one hidden layer, moves AUC by at most 0.003, far below the fold SD."],
    ["Integration (blue)", "Learned stack 0.890 ≈ MLP 0.889 ≈ simple average 0.889 > BN alone 0.876 on random folds."],
    ["Why", "A linear model already captures most of the signal (LR = MLP), so extra capacity has little to learn."],
  ];
  ab.forEach((a, i) => {
    const y = 1.35 + i * 1.82;
    card(s, 8.35, y, 4.35, 1.62);
    txt(s, a[0], { x: 8.6, y: y + 0.15, w: 3.9, h: 0.4, fontSize: 16, bold: true, fontFace: "Cambria" });
    txt(s, a[1], { x: 8.6, y: y + 0.55, w: 3.9, h: 1.0, fontSize: 14 });
  });
  who(s, "Akshith");
  s.addNotes("Presenter: Akshith. Ablations use fixed settings so that only one factor changes. None of the MLP regularisers individually matters at this sample size, and one hidden layer is as good as two. That does not mean regularisation is useless; it means the data, not the architecture, limit performance. The integration ablation shows the learned stack behaves almost like an average on random folds; its benefit appears only under hospital shift.");

  // ---------- 11. Explainability ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("What the MLP learns that the BN cannot", { placeholder: "title" });
  s.addImage({ path: "img/surface_top.png", x: 0.6, y: 1.35, w: 7.6, h: 3.35 });
  txt(s, "Predicted risk over age × maximum heart rate: MLP (left) is smooth; the BN (right) is a 3×3 step function from tertile bins.",
    { x: 0.6, y: 4.8, w: 7.6, h: 0.6, fontSize: 13, italic: true, color: C.accent5 });
  txt(s, [
    { text: "In the BN's DAG, blood pressure only connects to age, so once age is known it cannot change the BN's prediction", options: { bullet: true, breakLine: true } },
    { text: "The MLP learns some extra risk from blood pressure when chest pain is asymptomatic", options: { bullet: true } },
  ], { x: 0.6, y: 5.5, w: 7.6, h: 1.3, fontSize: 15, paraSpaceAfter: 6 });
  s.addChart(pres.charts.BAR, [{ name: "mean |SHAP|", labels: D.shap.labels, values: D.shap.vals }], Object.assign({
    x: 8.6, y: 1.35, w: 4.1, h: 5.4, barDir: "bar", chartColors: [HEX.accent1],
    showTitle: true, title: "MLP: top-6 mean |SHAP|", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: 0, valAxisMaxVal: 0.14, valAxisMajorUnit: 0.04, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valGridLine: { color: HEX.accent6, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
  }, chartText));
  who(s, "Hardik");
  s.addNotes("Presenter: Hardik. SHAP ranks ca, chest pain, ST depression, thal and slope highest, consistent with Phase 1. The surfaces show the architectural contrast: the BN discretises continuous features into tertiles and its fixed graph makes some features irrelevant once others are observed; the MLP learns smooth interactions. That complementarity is why we stack them. These explanations describe the fitted model on complete-case patients, not causes.");

  // ---------- 12. Conclusion ----------
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
  txt(s, "Conclusion and demo", { x: 0.8, y: 0.6, w: 11, h: 0.8, fontSize: 36, bold: true, color: C.background1, fontFace: "Cambria" });
  const concl = [
    [iconCheck, C.accent3, "Feedback met", "All four cohorts; TPE via OptunaSearchCV for all eight models; BN–MLP integration; IEEE paper."],
    [iconBalance, C.accent1, "No free lunch from DL", "MLP and stack tie LR and SVM on pooled data (AUC ≈ 0.89) and are well calibrated."],
    [iconHospital, C.accent2, "Shift dominates", "A held-out hospital costs 0.05–0.08 AUC; the stack is more robust than the MLP alone."],
    [iconWarn, C.accent4, "No clinical claim", "Old, referral-based, incomplete data. Not a diagnostic tool."],
  ];
  concl.forEach((c, i) => {
    const x = 0.8 + i * 3.0;
    s.addShape(pres.shapes.OVAL, { x, y: 1.6, w: 0.7, h: 0.7, fill: { color: c[1] }, line: { type: "none" } });
    s.addImage({ data: c[0], x: x + 0.17, y: 1.77, w: 0.36, h: 0.36 });
    txt(s, c[2], { x, y: 2.45, w: 2.7, h: 0.5, fontSize: 19, bold: true, color: C.background1, fontFace: "Cambria" });
    txt(s, c[3], { x, y: 2.95, w: 2.7, h: 1.6, fontSize: 15, color: HEX.accent6 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 4.75, w: 11.7, h: 1.95, fill: { color: C.text1 }, line: { color: C.accent5, width: 1 }, rectRadius: 0.08 });
  txt(s, "Live demo: Phase 2 Streamlit app", { x: 1.1, y: 4.9, w: 6, h: 0.45, fontSize: 19, bold: true, color: C.background1, fontFace: "Cambria" });
  txt(s, [
    { text: "Loads the saved Phase 2 run: pooled and per-cohort results, search evidence, ablations, calibration", options: { bullet: true, breakLine: true } },
    { text: "Risk estimator: all eight models, fitted once offline on 920 patients and cached; no search per click", options: { bullet: true } },
  ], { x: 1.1, y: 5.45, w: 5.6, h: 1.2, fontSize: 14, color: HEX.accent6, paraSpaceAfter: 4 });
  txt(s, [
    { text: "Demo models are labelled separately from the nested-CV estimates", options: { bullet: true, breakLine: true } },
    { text: "Input validation and the not-for-clinical-use disclaimer on every page", options: { bullet: true } },
  ], { x: 6.9, y: 5.45, w: 5.4, h: 1.2, fontSize: 14, color: HEX.accent6, paraSpaceAfter: 4 });
  txt(s, "Thank you. Questions?", { x: 8.3, y: 4.9, w: 4.0, h: 0.45, fontSize: 19, bold: true, color: C.background1, fontFace: "Cambria", align: "right" });
  who(s, "Kabir", true);
  s.addNotes("Presenter: Kabir. Summarise the four messages. Next steps: learn the BN structure, multiple imputation, per-site recalibration, more TPE trials and repeated outer folds. Then switch to the demo. In the app, the demo models are fitted once on all 920 patients and cached; their performance is the nested-CV estimate shown on the results page, not a new number. Every page keeps the not-for-clinical-use disclaimer.");

  await pres.writeFile({ fileName: "Phase2_Heart_Disease_Risk_Modelling.pptx" });
  await applyTheme("Phase2_Heart_Disease_Risk_Modelling.pptx", THEME);
  console.log("written");
})();
