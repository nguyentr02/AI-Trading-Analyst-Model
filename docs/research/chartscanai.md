# ChartScanAI: can `cryptoai/` learn from it or adopt it?

*Research note, 2026-10-05. Reviews the GitHub repository Omar-Karimov/ChartScanAI [1] and the literature on image-based chart prediction, to decide whether `cryptoai/` should adopt any of it. `cryptoai/` predicts P(up) for BTC/ETH/BNB/SOL vs USDT on Binance with a pooled HistGradientBoosting model on about 40 tabular features: 4h candles with a 6-candle (24h) horizon, and 1d candles with a 3-candle horizon. It uses walk-forward evaluation with an embargo, a 0.1% fee per side, and trades long-or-flat [14]. Citations are numbered; see [Sources](#sources). Companion to [neurotrader-methods.md](neurotrader-methods.md) and [swing-points-rolling-window-directional-change-pip.md](swing-points-rolling-window-directional-change-pip.md).*

**How this was researched.** I read every file in the repository: README, `app.py`, `requirements.txt`, `packages.txt`, `LICENSE` and the commit history. I also read all seven issue threads through the GitHub API. The repo publishes no training code, dataset or metrics. The training metadata in this note therefore comes from the weights file itself. I downloaded `weights/custom_yolov8.pt` and read its pickled metadata with Python's `pickletools`, without executing it. Those figures are labelled *(this note's check)* [2]. I read the full text of the paper the repo cites [6] and of Jiang, Kelly & Xiu [7]. For other studies I read abstracts or HTML versions; each one says which.

## Bottom line

- **ChartScanAI is a YOLOv8m object detector that draws "Buy" and "Sell" boxes on candlestick chart images.** Its author labelled the boxes by hand in Roboflow [1][3]. Inputs are 180-candle `mplfinance` charts rendered from yfinance data at 1h, 1d or 1wk [1]. The app is a Streamlit page: upload an image, get the boxes back [1].
- **The labels are subjective and drawn with hindsight.** The boxes sit on visible swing lows (Buy) and swing highs (Sell) inside the chart. A swing low is only recognisable once the candles to its right exist. The repo's own sample output has no box on the latest candles [1]. Users report that signals appear only "after they have fully formed" [3] and that the model "only renders retrospectively" [4]. The author's reply: the analysis "is inherently retrospective" and "is not designed to forecast future price movements in real time" [4].
- **The only metrics measure agreement with the labeller, not trading value.** At the end of training, the checkpoint reports validation precision 0.591, recall 0.604, mAP50 0.604 and mAP50-95 0.299 *(this note's check)* [2]. mAP measures how well predicted boxes overlap the hand-drawn ones [11]. The repo has no backtest, no out-of-sample return test and no fees. Asked "can you backtest on this model", the author said it is possible but did not report one [5].
- **The paper it builds on is weaker evidence than its abstract suggests.** Birogul et al. (2020) report "85%" success and "100% profit" [6]. The 85% is 11 profitable baskets out of 13. The profit is 102.68% over 376 days with commissions excluded. The test stocks were pre-selected "in terms of technical and basic analyses", and there was no benchmark [6].
- **The serious evidence for image models is Jiang, Kelly & Xiu (2023)** [7]. They trained a CNN on objective labels (the sign of the future return) using small black-and-white OHLC images of US stocks. Results:
  - Training data 1993–2000, test 2001–2019, model fixed.
  - Weekly long-short decile portfolios had out-of-sample gross Sharpe ratios of up to 7.2 equal-weight and 1.7 value-weight.
  - Net-of-cost Sharpe ratios were up to 4.0 weekly, 1.5 monthly and 0.9 quarterly (equal-weight).
  - That result is a **cross-sectional** sort over thousands of stocks. Our setting is four coins traded long-or-flat over time, which is a different problem.
- **Much of the CNN's edge in [7] came from how the image scales the data, not from the convolutions.** A plain logistic regression on the same OHLC numbers, rescaled to the window's high–low range as the image does, gave a "successful trading strategy" that was "generally inferior" to the CNN. Scaling the same numbers by cumulative return instead dropped one Sharpe ratio from 2.0 to 0.4 [7]. The CNN's simplest learned pattern is "closes at the low end of its recent high-low range → higher future return" [7]. That is the same idea as our existing `range_pos` feature [14].
- **Recommendation: do not adopt ChartScanAI (option d), and test the cheap version of the Jiang–Kelly–Xiu idea first.**
  - Add "image-scaled" OHLC features to the tabular model as **one pre-registered bundle**. Test them with walk-forward, fees and the selection-aware permutation test from [15].
  - Train a CNN (option b) only if that bundle shows lift. On a CPU-only machine, a proper permutation test of a CNN is not affordable.
  - For the dashboard, mark confirmed swing points with a deterministic rule rather than YOLO. It is objective, adds no dependencies, and avoids AGPL licensing.

---

## 1. What ChartScanAI is

### 1.1 Repository facts

| Item | Value |
|---|---|
| Files | `README.md`, `app.py` (Streamlit app), `requirements.txt`, `packages.txt` (`libgl1`), `LICENSE`, `images/`, `weights/custom_yolov8.pt` (52 MB) [1] |
| Activity | Created 2024-06-22; last push 2024-06-29. About 15 commits, all between 2024-06-27 and 2024-06-29 [1] |
| Licence | MIT for the repository [1]. The weights file's own metadata reads `AGPL-3.0 (https://ultralytics.com/license)` [2] (see §2.5) |
| Dependencies | `ultralytics==8.2.39`, `torch==2.3.1`, `streamlit==1.36.0`, `yfinance==0.2.40`, `mplfinance==0.12.10b0`, `matplotlib`, `pandas`, `numpy`, `pillow` [1] |
| Training code, notebooks, dataset, `data_custom.yaml` | **Not published.** There is no Roboflow dataset link in the repo [1] |
| Inspiration | Birogul, Temür & Kose, IEEE Access 2020, which used YOLOv3 on Borsa İstanbul charts [1][6] |

### 1.2 Input

`generate_chart()` downloads a yfinance ticker at 1d, 1h (last 730 days) or 1wk. It keeps the **latest 180 candles** and plots them with `mplfinance` (`type="candle"`, `style="yahoo"`, `axisoff=True`, no volume, a title, `figsize=(18, 6.5)`, `dpi=100`), which gives 1800×650 px [1]. That size matches the 1800 × 650 charts in Birogul et al. [6]. The detector runs on any uploaded image. The confidence slider defaults to 0.30 [1].

According to the author, the training images were "multiple cryptocurrencies and stock pairs across various timeframes (both low and high)". All had exactly 180 candles and were annotated "primarily through manual annotation with Roboflow" [3]. The number of images and labels is not disclosed anywhere I could find [1][3].

### 1.3 Model and training *(this note's check, from checkpoint metadata [2])*

| Field | Value |
|---|---|
| Classes | `{0: 'Buy', 1: 'Sell'}` |
| Base model | `yolov8m.pt` (COCO-pretrained YOLOv8 medium: 25.9 M parameters, 78.9 GFLOPs at 640 px [12]) |
| Data config | `data_custom.yaml` (not in the repo) |
| Image size / batch / requested epochs | 640 / 8 / 150 (patience 100) |
| Epochs actually logged | 83 (the reason is not stated) |
| Augmentation | `fliplr=0.5` (horizontal flip), `mosaic=1.0`, `translate=0.1`, `scale=0.5`, `hsv_h=0.015` |
| Training date / library version | 2024-06-25 / Ultralytics 8.2.39 |
| Final validation metrics | precision 0.591, recall 0.604, **mAP50 0.604, mAP50-95 0.299**, fitness 0.329 |
| Best mAP50 over logged epochs | 0.608 (epoch 74) |

How the validation split was made (random or by time, and by asset) is not recorded in the repo.

### 1.4 What the output looks like

The README's sample `out3.jpg` shows ETH-USD over 180 candles. It has three Buy boxes (confidence 0.34–0.51) on local lows and two Sell boxes (0.51, 0.61) on local highs. **None of them is on the last candles** [1]. In Issue #7 (2025-08-27, unanswered), a user asks how to tell whether the last 5 candles contain a Buy signal and says reading it from the image was "not very successful" [1].

---

## 2. Critical evaluation

### 2.1 The labelling is subjective

- **ChartScanAI.** The labels are boxes drawn by one person in Roboflow [3]. The repo gives no written rule for what makes a "Buy", so the labels cannot be reproduced.
- **Birogul et al.** Their Buy boxes mark "zone 1" (the buying zone at a bottom) and their Sell boxes mark "zones 3 and 4" (topping zones). They labelled 550 annual BIST charts by hand in LabelImg: 10,009 labels, of which 5,161 Buy and 4,848 Sell. They "considered" the Dow theory, Japanese candlestick formations and technical indicators [6]. This is closer to a rule, but it is still a human judgement made while looking at the whole year, including what came after each box.

### 2.2 The metrics measure agreement with the labeller, not trading value

- **What mAP measures.** mAP50 is mean average precision at a box-overlap (IoU) threshold of 0.50 against the ground-truth boxes [11]. A perfect mAP would mean the model draws the same boxes as the annotator. It says nothing about whether price rises after a Buy box.
- **What the ChartScanAI numbers mean.** A mAP50 of 0.60 means the model agrees only moderately with its own labeller, before any question of profit [2].
- **What is missing.** There is no hit-rate on future returns, no backtest, no out-of-sample period and no fees [1][5].

### 2.3 Look-ahead is built into the task

- **Each label needs the bars after it.** A swing low is defined by higher prices on *both* sides. In a training image the annotator sees the candles to the right of the box, and so does the CNN. The detector learns "this is a confirmed bottom", which is the same lag problem as the rolling-window extrema in the swing-point note [16].
- **Users and the author describe the result.** Live charts get few or no boxes at the right edge [3][4]. A tester who examined more than 500 ETH/USDT 5m and 1h images found some boxes "approximately some bars away from the latest bar" [3]. The author's advice was to lower the confidence threshold to about 0.25–0.30 to get "earlier signals", which "may produce more false positives" [3].
- **Image scaling also uses the whole window.** Each chart's vertical axis is scaled to the 180-candle window, so where a candle sits in the image depends on prices that came later.
- **The training augmentation flips charts left–right.** With `fliplr=0.5` [2], half the training images run backwards in time. A V-shaped bottom flipped is still a V-shaped bottom, so this is harmless for recognising shapes. It confirms, though, that the model is taught time-symmetric shapes, not anything about which way price goes next.

### 2.4 No trading evidence, and the cited paper's evidence is weak

- **ChartScanAI** has no backtest [1][5].
- **Birogul et al.** traded "1-year chart images of the most appropriate stocks", chosen after "technical and basic analyses" of each stock [6]. They bought on a Buy box at the last candle and held until a Sell box appeared. Over 376 days, 13 baskets and 112 round trips, they report $10,000 → $20,267 (+102.68%) with **commissions excluded**, and 11 of 13 baskets profitable ("84.6%") [6].
- **What that test lacks:**
  - It covers one year in one market.
  - It has no benchmark index return.
  - It has no significance test.
  - The stock pre-selection is discretionary.
- **A newer follow-up** tested YOLO v3/v8/v9/v11 with moving averages added to the chart [9]. It reports only detection metrics (F1 up to 0.06 higher, recall up to 0.18 higher than prior work), with no trading returns, according to the abstract [9].

### 2.5 Licence

The repository is MIT [1], but the weights are a fine-tune of Ultralytics `yolov8m.pt` and their metadata states AGPL-3.0 [2]. Ultralytics' position is that "all Ultralytics YOLO trained models fall under the AGPL-3.0 License by default". It also says that using its code or trained models requires either open-sourcing the whole project under AGPL-3.0 or buying an Enterprise licence, including for SaaS/API deployment [10]. This is Ultralytics' own interpretation, not legal advice. Even so, shipping the weights inside our dashboard would probably bring AGPL obligations for the dashboard code.

---

## 3. What the literature says about image-based price prediction

### 3.1 Jiang, Kelly & Xiu (2023), *Journal of Finance* [7]

- **Data and labels.**
  - Daily US stocks (CRSP), 1993–2019.
  - Each image is a black-and-white OHLC bar chart of the past 5, 20 or 60 days, three pixels wide per day, with a moving-average line and volume bars.
  - The vertical axis is scaled so that the window's maximum and minimum fill the image.
  - The label is 1 if the return over the next 5, 20 or 60 days is positive. That gives nine models, each retrained five times and averaged [7].
- **Validation.**
  - The CNN is trained once on 1993–2000 (a random 70/30 train/validation split) and **held fixed** for the 2001–2019 test period [7].
  - Model sizes are 155,138 parameters (5-day), 708,866 (20-day) and 2,952,962 (60-day) [7].
- **Results (out of sample).**
  - Weekly equal-weight long-short decile Sharpe ratios are 7.2, 6.8 and 4.9 for the 5-, 20- and 60-day image models.
  - Value-weight Sharpe ratios are 1.4–1.7.
  - The best benchmark trend signals (TREND and WSTR) reach 2.9 and 2.8 equal-weight [7].
  - The monthly and quarterly strategies reach up to 2.4 and 1.3 [7].
- **Costs.**
  - With 10–20 bp trading costs, net Sharpe ratios are up to 4.0 (weekly), 1.5 (monthly) and 0.9 (quarterly), equal-weight [7].
  - The authors say the weekly strategy "is mostly accessible to investors who behave as market makers" [7].
  - Restricted to the largest 500 stocks, Sharpe ratios are still above 1.0 [7].
- **What the CNN learns.**
  - It correlates with known signals such as reversal, size, dollar volume and illiquidity, but these "explain only about 10%" of the variation in its forecasts [7].
  - One simple approximation of a pattern it detects: "when a stock closes on the low end of its recent high-low range, future returns tend to be high" [7].
- **Image versus numbers.**
  - A logistic regression on the same OHLC data, scaled the way the image scales it, is "generally inferior to the full nonlinear CNN" but still beats the benchmarks, and at 60-day horizons it sometimes beats the CNN [7].
  - The same logistic model with prices scaled by cumulative return is "substantially weaker". The example given is an equal-weight Sharpe of 2.0 falling to 0.4 [7].
  - A 1D CNN on the raw time series was also tested. Its performance depended strongly on how the inputs were scaled [7].
  - The authors concede that "a well-crafted time-series model … may outperform the CNN" [7].

**Relevance to us.** The finding is about ranking thousands of stocks at once, it is strongest at a one-week horizon, and it is concentrated in small and illiquid stocks (equal-weight Sharpe ratios are more than double the value-weight ones [7]). With four highly correlated coins traded long-or-flat, we cannot form deciles. The time-series version of the claim, "this coin's image predicts this coin's next 24h", is not what they tested. The lesson that transfers is **how the inputs are scaled**: normalise each bar's OHLC to the window's high–low range.

### 3.2 Crypto-specific studies

- **Haggett (2026), arXiv preprint, single author [8].**
  - Setup: candlestick, Gramian-Angular-Field and multi-channel images with CNN, ResNet18, EfficientNet-B0 and ViT models. Daily BTC, ETH and SPY data, 2018–2024. The label is whether the 7-day forward return exceeds 2%. The split is chronological 70/15/15.
  - Results: the headline AUC of 0.892 comes from one BTC configuration (128×128 px). The baseline BTC AUC is 0.734 and ETH's is 0.492 [8].
  - Weaknesses:
    - There are about 500 samples per experiment, so each test set is about 75 overlapping 7-day labels.
    - There are no confidence intervals or multiple seeds.
    - The paper says its "evaluation focuses on discrimination ability rather than risk-adjusted returns after accounting for transaction costs" [8].
    - There is no tabular baseline.
  - I treat it as anecdotal (read via the arXiv HTML version).
- **Other crypto chart-image papers.** I found a few in conference proceedings and book chapters (e.g., Bitcoin candlestick-pattern recognition), but did not find one with a walk-forward test, a permutation test and after-fee returns on Binance majors at 4h/1d. I could not access an ETF study (*European Journal of Finance*) or a Korean-market replication (ScienceDirect, 2025) beyond their titles, so I do not cite their results.

### 3.3 Chart patterns more generally

Lo, Mamaysky & Wang (2000) defined chart patterns algorithmically with kernel regression, including a confirmation delay. They found that the patterns change the conditional return distribution of US stocks, but modestly [13]. The swing-point note covers this literature and the lag issue [16].

---

## 4. Options for `cryptoai/`

Context from our repo: about 17,000 4h candles per coin since 2019 (SOL about 13,500) and about 2,800 daily candles (SOL about 2,250). The repo has no PyTorch dependency and runs on a 12-core CPU. One 4h walk-forward takes about 9 s. The null AUC sd from the permutation test is 0.0085 (4h) and 0.015 (1d) [14][15].

| | (a) ChartScanAI weights as a feature | (b) Our own JKX-style CNN on future-return labels | (b′) "Image-scaled" tabular features (no CNN) | (c) YOLO boxes on dashboard charts only | (d) Skip |
|---|---|---|---|---|---|
| **What** | Render a 180-candle chart ending at bar *t* in the same mplfinance style. Feature = max Buy/Sell confidence of boxes touching the last *k* candles | Render a 20-bar (and/or 5-bar) OHLC+MA+volume image per coin per bar. Train a small CNN for P(up over horizon). Feed its out-of-sample probability to HGB | For the last 5 bars, compute (O,H,L,C − lowₙ)/(highₙ − lowₙ) over the window. Add MA and volume scaled the same way. Add as about 25 HGB features | Run the detector on the dashboard chart and overlay boxes. No model input | — |
| **Effort** | 1–2 days: torch + ultralytics + matching renderer | 1–2 weeks: renderer, CNN, nested walk-forward, stacking | About half a day in `features.py` | About 1 day | 0 |
| **Compute (CPU)** | YOLOv8m CPU ONNX is about 235 ms per image (Ultralytics benchmark, hardware not stated) [12]. Plus rendering. Backfilling about 65k 4h bars ≈ several CPU-hours *(estimate)* | Training 8 folds × 5 seeds on about 65k small images is likely hours per run *(estimate)*. A 100-permutation MCPT would take days | Seconds. Full MCPT as today (about 40 min for N = 200 on 4h [15]) | About 0.3 s per refresh; fine | — |
| **Leakage risks** | Weights trained on unknown crypto charts up to 2024-06-25 [2], so every bar before that date is contaminated. Only about 15 months are clean. Renderer mismatch | Overlapping labels (embargo needed). The CNN must be fitted inside each fold. HGB needs **out-of-fold** CNN outputs for its own training rows (cross-fitting). Pixel normalisation from training data only [7] | Low: uses bars ≤ *t* only. Same embargo as now | None for the model. **User risk:** boxes on past swings look like signals but repaint/lag [3][4] | — |
| **Expected value** | Very low: it encodes "a confirmed swing *k* bars ago", which a rolling-window rule gives for free [16]. Subjective labels, mAP 0.60 | Unknown and probably small. Four coins give a time-series problem, not JKX's cross-section. Small sample, especially 1d | Small but cheap. It is the part of [7] that a linear model already captures, though `range_pos` covers some of it | Cosmetic. AGPL exposure [10]. Torch install (~GBs) for a picture | Saves time for higher-ranked ideas [15] |
| **How to test fairly** | Out of sample only after 2024-06-25. Walk-forward AUC and net Sharpe vs baseline on the same folds. Expect the null sd to exceed 0.0085 on this short window | Nested walk-forward. Compare against (b′) and the baseline on the same folds. Permutation test with as many permutations as affordable (p-value resolution will be coarse) | One pre-registered bundle. Same 8-fold walk-forward. Selection-aware MCPT with N ≥ 200 [15]. Net Sharpe at 0.1%/side with fixed thresholds 0.55/0.48 | Not a model change, so no statistical test. Label the boxes "confirmed after k bars" | — |

### 4.1 Recommendation

1. **Do not adopt ChartScanAI's weights or app (d for (a)).**
   - Its labels are subjective and use hindsight.
   - Its only metric is agreement with its labeller.
   - It has no trading evidence.
   - Its weights may have seen our test period.
   - It carries AGPL obligations.
2. **Run (b′) once.** This is the only part worth borrowing from the image literature right now. Spec, fixed before seeing results:
   - **Window and features.** Window *n* = 5 bars. Features are the close, high, low and open of each of the last 5 bars scaled to [low₅, high₅] (20 features). Add the 20-bar EMA position scaled the same way (1 feature), and the volume of the last 5 bars divided by the window's maximum volume (5 features). Use the same definition on 4h and 1d.
   - **Model.** Add the bundle to the current feature set and do not change the HGB hyperparameters.
   - **Acceptance.** Pooled walk-forward AUC must beat the maximum-over-tried-feature-sets null at p < 0.05 on 4h with N ≥ 200 permutations [15]. Mean net Sharpe must not fall. Report 1d but do not use it to decide, since 1d already fails the permutation test.
   - **No retries.** If it fails, stop. Do not tune *n* on the test output.
3. **Revisit (b) only if (b′) passes** and we are willing to add PyTorch. Even then, run the CNN offline (weekly), not in the always-on per-candle service.
4. **For the dashboard (instead of (c)),** draw swing highs and lows confirmed by the rolling-window rule from the swing-point note [16]. Show the confirmation delay explicitly, for example "low at bar *i*, confirmed at *i + k*". This gives the same visual help with no ML, no torch, no AGPL, and is honest about the lag.

---

## Results in this repo (2026-10-05)

Option b′ was tested once, as a fixed bundle with no tuning afterwards: 25 inputs, the open, high, low and close
of each of the last 5 candles rescaled to the 20-candle high–low range, plus their volume relative to the
window's maximum volume. The test used walk-forward evaluation on 4h candles, with the models' other inputs
(including the new 15m/1h patterns) unchanged.

| Model | AUC without | AUC with | Mean Sharpe without → with | Years better |
|---|---|---|---|---|
| Next 4 hours (`4h_next`) | 0.5461 | 0.5471 | −0.16 → −0.16 | 4 of 6 |
| Next 1 day (`4h`) | 0.5334 | 0.5359 | 0.69 → 0.59 | 4 of 6 |

**Not adopted.** The AUC gains are within noise, the year-by-year results are mixed, and the next-1-day
backtest got worse. Per the plan above, there is no tuning and no CNN.

## Caveats

- **The training metadata comes from the pickled checkpoint**, read without unpickling it. It reflects whatever Ultralytics stored at the end of training. The number of training images and the validation split could not be determined [2].
- **The dataset composition** ("multiple cryptocurrencies and stock pairs", various timeframes) is only the author's statement in an issue thread [3]. I could not verify which assets or dates were used, so the contamination window in option (a) is a conservative assumption.
- **Compute figures marked *(estimate)* are mine and were not measured.** The YOLOv8m CPU latency is Ultralytics' COCO benchmark at 640 px on unstated hardware [12].
- **Jiang, Kelly & Xiu's** numbers are for US equities. I quote only figures that appear in the published text. I did not read the Internet Appendix.
- **The arXiv crypto study [8]** is a single-author preprint that is not peer reviewed. Its details come from the arXiv HTML version.
- **The AGPL reading** is Ultralytics' own [10] and is not legal advice.

## Sources

1. Omar-Karimov, ChartScanAI, GitHub repository (README, `app.py`, `requirements.txt`, `packages.txt`, `LICENSE`, `images/out3.jpg`, commit history, Issue #7), accessed 2026-10-05. https://github.com/Omar-Karimov/ChartScanAI
2. *(this note's check)* Metadata of `weights/custom_yolov8.pt` (`best/data.pkl`: `train_args`, `train_metrics`, `train_results`, `date`, `version`, `license`, `names`), read with Python `pickletools`. https://github.com/Omar-Karimov/ChartScanAI/blob/main/weights/custom_yolov8.pt
3. ChartScanAI Issue #2, "What dataset was your model trained on?" (codenong and Omar-Karimov, 2025-01-30). https://github.com/Omar-Karimov/ChartScanAI/issues/2
4. ChartScanAI Issue #3, "Repaint" (DSTGlobal; reply by Omar-Karimov, 2025-05-06). https://github.com/Omar-Karimov/ChartScanAI/issues/3
5. ChartScanAI Issue #1, "can you backtest on this model" (reply 2024-12-10). https://github.com/Omar-Karimov/ChartScanAI/issues/1
6. Birogul, S., Temür, G. & Kose, U. (2020). YOLO Object Recognition Algorithm and "Buy-Sell Decision" Model Over 2D Candlestick Charts. *IEEE Access* 8, 91894–91915. https://doi.org/10.1109/ACCESS.2020.2994282 ; full text via Düzce University repository: https://acikerisim.duzce.edu.tr/items/84f3408b-ffa3-4f20-a5b1-61b2040c51ff
7. Jiang, J., Kelly, B. & Xiu, D. (2023). (Re-)Imag(in)ing Price Trends. *Journal of Finance* 78(6), 3193–3249. https://doi.org/10.1111/jofi.13268 ; PDF hosted by Yale Economics: https://economics.yale.edu/research/re-imagining-price-trends
8. Haggett, D. M. (2026). Visual Chart Representations for Cryptocurrency Regime Prediction: A Systematic Deep Learning Study. arXiv:2605.00875 (preprint). https://arxiv.org/abs/2605.00875
9. Santos, M. A. S., da Silva, A. R. & Ortoncelli, A. R. (2025). YOLO-Based Detection of Buy and Sell Signals in Candlestick Charts with Moving Averages. *Anais do ENIAC 2025*. https://sol.sbc.org.br/index.php/eniac/article/view/38736 (abstract only)
10. Ultralytics, "Licensing" (AGPL-3.0 and Enterprise; FAQ on trained models). https://www.ultralytics.com/license
11. Ultralytics Docs, "Performance Metrics Deep Dive" (definitions of precision, recall, IoU, mAP50, mAP50-95). https://docs.ultralytics.com/guides/yolo-performance-metrics/
12. Ultralytics Docs, "YOLOv8" (COCO detection table: YOLOv8m 25.9 M params, 78.9 B FLOPs, CPU ONNX 234.7 ms at 640 px). https://docs.ultralytics.com/models/yolov8/
13. Lo, A., Mamaysky, H. & Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation. *Journal of Finance* 55(4), 1705–1765. https://doi.org/10.1111/0022-1082.00265
14. This repo: `cryptoai/features.py` (`build`, `range_pos`, `target`), `cryptoai/model.py` (`walk_forward`, `_new_model`), `cryptoai/config.py` (`HORIZON`, `FEE`, `ENTER_PROB`, `EXIT_PROB`, `MIN_AUC`), `data/*_4h.csv`, `data/*_1d.csv`.
15. This repo: [neurotrader-methods.md](neurotrader-methods.md), §1.7 (prototype walk-forward permutation test on `cryptoai`; null AUC sd and timings) and §9.1 (selection-aware MCPT).
16. This repo: [swing-points-rolling-window-directional-change-pip.md](swing-points-rolling-window-directional-change-pip.md) (rolling-window extrema, confirmation lag, causality checks).
