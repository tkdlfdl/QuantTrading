# Strategy Brief: Reddit Sentiment Moving-Average

*Research analyst brief — compiled 2026-08-15. All citations below were verified against primary or authoritative secondary sources (journal pages, SSRN, RePEc, publisher PDFs). Where a specific figure could not be extracted from a primary source, it is flagged as unverified rather than reported as fact.*

---

## Hypothesis

**Signal proposed:** Scrape Reddit posts about stocks, run sentiment analysis, compute a **moving average (smoothed) of the sentiment index** over time, and go **long when smoothed sentiment is high / rising** (analogously short or flat when low).

**What the literature must support for this to be tradable:**
1. That social-media/message-board sentiment carries information about *future* (not just contemporaneous) returns.
2. That **smoothing/averaging** the sentiment series improves — rather than destroys — the signal.
3. That the effect is **long-following (momentum)** rather than **contrarian/reversal** (which would flip the sign of the proposed rule).
4. That it survives **transaction costs** and **out-of-sample / post-publication decay**.

**Preview of verdict:** The peer-reviewed evidence is **mixed-to-skeptical** for a naive "buy-the-hype" long. It *does* support (a) that smoothing/aggregating sentiment lengthens the predictive horizon (Heston–Sinha; RavenPack practice), and (b) short-horizon predictability from curated social sentiment. But the dominant finding for **anonymous Reddit/message-board sentiment specifically** is **short-run momentum followed by long-run reversal**, a **contrarian sign at the aggregate level**, sub-transaction-cost magnitudes, and a **post-GameStop regime break** that eliminated predictability. This argues for a **contrarian / capitulation-buying** framing over naive hype-chasing.

---

## Literature (tiered)

### Tier S — Journal of Finance / JFE / Review of Financial Studies

**Antweiler & Frank (2004), "Is All That Talk Just Noise? The Information Content of Internet Stock Message Boards," *Journal of Finance* 59(3), 1259–1294. [JF — S]**
- **Sample/universe:** Full-year 2000; 45 firms (DJIA + Dow Jones Internet Commerce Index), US large-caps.
- **Sentiment source/method:** 1,559,621 Yahoo! Finance + Raging Bull messages; **Naïve Bayes** (+ SVM robustness), trained on 1,000 hand-coded buy/hold/sell messages. Aggregated to a per-window **bullishness** index and a **disagreement** index (period aggregation, not a moving average).
- **Result:** Small **negative** next-day return predictability after controlling for bid-ask bounce — statistically significant but **"economically very small in comparison to plausible transaction costs."** Message activity **predicts volatility** (strongest result); disagreement predicts trading volume.
- **Caveat:** Explicitly **NOT tradable after costs**. Return effect is the weakest of the three findings.

**Tetlock (2007), "Giving Content to Investor Sentiment: The Role of Media in the Stock Market," *Journal of Finance* 62(3), 1139–1168. [JF — S] (Smith–Breeden Prize)**
- **Sample/universe:** Daily, 1984-01-02 to 1999-09-17; aggregate US market (DJIA), plus SMB and detrended NYSE volume.
- **Sentiment source/method:** WSJ "Abreast of the Market" column; General Inquirer / Harvard-IV-4 dictionary → PCA "**pessimism factor**" used in a VAR.
- **Smoothing:** Sentiment factor is a daily series (not moving-averaged); MA-detrending is applied only to the *volatility control*, not the sentiment measure.
- **Result:** High media pessimism → **−8.1 bp** on next-day DJIA return (t = 3.94), then **full reversal within the trading week** (lags 2–5 reversal ≈ +6.8 bp). Extreme pessimism → high volume.
- **Caveat:** Effect is **temporary, mean-reverting** ("reversion to fundamentals"), consistent with noise-trader models. No after-cost strategy claimed; the return effect is transient.

**Chen, De, Hu & Hwang (2014), "Wisdom of Crowds: The Value of Stock Opinions Transmitted Through Social Media," *Review of Financial Studies* 27(5), 1367–1403. [RFS — S]**
- **Sample/universe:** 2005–2012; 97,070 single-ticker **Seeking Alpha** articles + reader comments, 7,000+ US firms.
- **Method:** Fraction of negative words (Loughran-McDonald / Tetlock tradition). Cross-sectional, **~3-month holding horizon** (not smoothed input).
- **Result:** More negative sentiment → **negative future returns over ~3 months**. Bottom-minus-top long/short on article negativity ≈ **2.6 bp/day** (t = 2.87); comment-based ≈ 2.2 bp/day. Also predicts earnings surprises, incremental to analyst reports and news.
- **Caveat:** ~2.4–2.6 bp/day on the raw spread (~6.5% annualized), modest per-day; robust to analyst/news controls but **no net-of-cost Sharpe reported**. Note: **curated** analysis (Seeking Alpha) is stronger than anonymous chat boards.

**Da, Engelberg & Gao (2015), "The Sum of All FEARS: Investor Sentiment and Asset Prices," *Review of Financial Studies* 28(1), 1–32. [RFS — S] (Editor's Choice)**
- **Sample/universe:** 2004–2011 daily; S&P 500, VIX, realized vol, mutual-fund flows; cross-sectional tests on beta/vol-sorted stocks.
- **Method:** **FEARS index** = daily **average** of standardized Google Trends search-volume changes for 30 dynamically-selected sentiment terms (a cross-sectional averaging/smoothing across terms).
- **Result — REVERSAL:** +1 SD FEARS → **−20 bps** same-day S&P 500 return, **"almost completely reversed after two days"** (t+1, t+2 significant at 1%; insignificant by t+3–t+5). Strongest in high-beta / high-vol / high-downside-risk (hard-to-arbitrage) stocks. Temporary volatility spike; next-day equity-fund outflows ("flight to safety").
- **Caveat:** Authors note Google releases SVI with a **one-day delay** → "predictive regressions cannot be run in real time." ~20 bps is small vs round-trip costs; **no after-cost Sharpe** claimed.

**Huang, Jiang, Tu & Zhou (2015), "Investor Sentiment Aligned: A Powerful Predictor of Stock Returns," *Review of Financial Studies* 28(3), 791–837. [RFS — S]**
- **Method:** **PLS** denoising of Baker-Wurgler proxies → "aligned" sentiment index (statistical smoothing, not a moving average).
- **Result:** Much stronger in- and out-of-sample prediction of the **aggregate market** than raw BW; **contrarian sign** (high sentiment → lower future market returns). Monthly frequency.

**Baker & Wurgler (2006), "Investor Sentiment and the Cross-Section of Stock Returns," *Journal of Finance* 61(4), 1645–1680. [JF — S]**
- **Method:** First PCA of six market-based proxies (closed-end fund discount, turnover, IPO count/first-day returns, equity share, dividend premium), orthogonalized to macro.
- **Result:** **Cross-sectional contrarian** — low sentiment → high subsequent returns on speculative/hard-to-value stocks; high sentiment → underperformance of those stocks. Level of an index, not a moving-average signal; not market-based social data.

**Cookson & Niessner (2020), "Why Don't We Agree? Evidence from a Social Network of Investors," *Journal of Finance* 75(1), 173–228. [JF — S]**
- **Sample:** StockTwits, main window 2013–2014; 1,442,051 posts, 12,029 users, 9,755 tickers.
- **Result:** This is a **disagreement/volume** paper, not a return-predictability paper. Within-group disagreement links to ~2.5–4× the trading volume of cross-group; explains ~1/3 of earnings-announcement volume. **No return Sharpe/alpha**; no smoothing.
- *(Note: a related Cookson-Niessner-type "The Social Signal" appears in JFE 2024 — common sentiment predicts modest positive next-day abnormal returns, common attention negative; effects small. Metadata verified via search only, full text not fetched — treat authorship/year as high-confidence-unverified.)*

**Bradley, Hanousek Jr., Jame & Xiao (2024), "Place Your Bets? The Value of Investment Research on Reddit's Wallstreetbets," *Review of Financial Studies* 37(5), 1409–1459. [RFS — S] (SSRN working paper 2021)**
- **Sample/universe:** ~2018–2021, split **pre- vs post-GameStop (Jan 2021)**; US equities in WSB "Due Diligence" posts.
- **Result (verbatim abstract):** *"Before the GameStop short squeeze, recommendations are significant predictors of returns and cash-flow news. This predictability is eliminated post-GME."* Post-GME, reports shift toward price-pressure/attention-grabbing stocks; retail trade informativeness strong pre-GME, absent post-GME.
- **Caveat:** Core message is a **regime break** — WSB DD lost return-predictive value in exactly the modern meme-era regime a live book would operate in. Exact CAR/alpha magnitudes were paywalled and **could not be extracted** — do not cite a specific number.

### Tier A — JBF / Accounting Review / field-top

**Renault (2017), "Intraday online investor sentiment and return patterns in the U.S. stock market," *Journal of Banking & Finance* 84, 25–40. [JBF — A]**
- **Sample/universe:** 2012–2016; SPY (index-level); ~60M StockTwits messages.
- **Method:** Custom StockTwits lexicon (beats standard dictionaries); sentiment **aggregated into half-hour interval buckets**.
- **Result:** The **first half-hour change in sentiment predicts the last half-hour SPY return** (same-day intraday), driven by **novice/noise traders**.
- **Caveat:** Index-level intraday predictability, not a costed cross-sectional alpha; interval aggregation, not a long moving average.

**Bartov, Faurel & Mohanram (2018), "Can Twitter Help Predict Firm-Level Earnings and Stock Returns?" *The Accounting Review* 93(3), 25–57. [Accounting Review — A]**
- **Sample/universe:** 2009–2012; firm-level tweets in a **pre-earnings-announcement window**.
- **Result:** Aggregate pre-announcement Twitter opinion **predicts the quarterly earnings surprise AND the announcement-window abnormal returns**; stronger for weaker information environments. Survives concurrent-media controls. No free tradable Sharpe stated.

**Sun, Najand & Shen (2016), "Stock Return Predictability and Investor Sentiment: A High-Frequency Perspective," *Journal of Banking & Finance* 73, 147–164. [JBF — A]**
- **Sample/universe:** Intraday (half-hour) S&P 500 + ETFs; high-frequency sentiment from news wires + internet + social media.
- **Result:** **Lagged half-hour sentiment predicts intraday returns** (persists into the last ~2 hours), distinct from intraday momentum; authors claim "significant economic value" in market-timing. Net-of-cost Sharpe not disclosed on accessible pages.

**Jiang, Lee, Martin & Zhou (2019), "Manager Sentiment and Stock Returns," *Journal of Financial Economics* 132(1), 126–149. [JFE — S/A+]**
- **Method:** Averaged tone index from 10-K/10-Q + conference-call text (disclosure tone, not social media), monthly.
- **Result:** Strong **negative** (contrarian) predictor of aggregate returns — in-sample R² 9.75%, OOS R² 8.38%; strongest in hard-to-value / costly-to-arbitrage firms.

### Practitioner / Financial Analysts Journal

**Heston & Sinha (2017), "News vs. Sentiment: Predicting Stock Returns from News Stories," *Financial Analysts Journal* 73(3), 67–83. [FAJ — Practitioner, top] — *most directly relevant to the smoothing question***
- **Sample:** >900,000 news stories, US equities; Thomson Reuters neural-net sentiment.
- **Smoothing — central:** Daily news predicts only 1–2 days, but **weekly-aggregated (time-averaged) sentiment predicts returns for a full quarter (~13 weeks).** This is the canonical result that **smoothing/aggregating the sentiment series extends the predictive horizon.**
- **Result:** Positive news → fast response; negative news → long delayed drift resolving near next earnings. FAJ abstract does not disclose a net long/short Sharpe — treat the horizon finding as the load-bearing claim.

**RavenPack, "Constructing a Sentiment Factor" (vendor white paper — NOT peer-reviewed)**
- **Explicit moving average:** The RavenPack Sentiment Index is defined as *"the **90-day simple moving average** of the difference between the count of positive and negative news."* Long/short on "news beta" against this smoothed index.
- **Caveat:** Vendor material — no sample period, Sharpe, or turnover/cost disclosed. Evidence of *industry practice of smoothing sentiment with a rolling MA*, not peer-reviewed performance.

### Reddit / meme-stock (peer-reviewed, mostly field/specialist journals)

**Reichenbach & Walther (2023), "Financial recommendations on Reddit, stock returns and cumulative prospect theory," *Digital Finance* 5(2), 421–448. [specialist — mid] — *closest analog to a Reddit long-only book***
- **Sample:** 2008 to 2022-08-31; US stocks from r/wallstreetbets, r/stocks, r/investing; VADER + Loughran-McDonald sentiment; portfolio weighted by posts-per-day (attention weighting).
- **Result:** Buying recommended stocks → higher raw return but **UNFAVORABLE Sharpe (worse than market risk-adjusted)**; positive-but-insignificant 1–3-month alphas turning **significantly NEGATIVE at ~1 year** — classic **short-run inflation, long-run reversal**.

**Huang & Nolan (2025), "Dumb Money? Social Network Attention Herding, Sentiment, and Markets," *Journal of Finance and Data Science*. [specialist]**
- **Method:** LLM-classified WSB sentiment + individual-stock attention/herding; monthly portfolios.
- **Result:** WSB sentiment is a **CONTRARIAN** predictor (bullish herding → lower future returns); "Attention Herding Portfolio" generates sizable alphas; no return reversal after high-engagement herding. Exact magnitudes not extractable (pages blocked).

**Fernandez-Perez, Indriawan & Khomyn (2025), "Emotions and stock returns during the GameStop bubble," *Financial Review* 60(3), 1063–1084.** — Reddit NRC-EmoLex emotions; joy predicts before peak, fear at peak, anger after burst; fear predicts intraday GME returns. Within-event study, not a cross-sectional signal.

**Hasso, Müller, Pelster & Warkulat (2022), "Who participated in the GameStop frenzy?" *Finance Research Letters* 45, 102140.** — Brokerage-account evidence: GME participants had prior lottery/speculative-trading histories; late buyers lost money. Behavioral, not a sentiment-prediction paper.

### Skeptical / replication-failure (critical for risk assessment)

**Kim & Kim (2014), "Investor Sentiment from Internet Message Postings and the Predictability of Stock Returns," *Journal of Economic Behavior & Organization*. [field]**
- **Sample:** 2005–2010; 91 firms; **>32M Yahoo! Finance messages** (self-tags + ML). Larger/longer than Antweiler-Frank.
- **Result — NULL:** **No evidence** message-board sentiment forecasts returns, volatility, or volume. Sentiment is **driven by past returns** (sentiment follows price, not vice versa). Key robustness counterweight to Antweiler-Frank.

**Lachanski & Pav (2017), "Shy of the Character Limit: 'Twitter Mood Predicts the Stock Market' Revisited," *Econ Journal Watch* 14(3). [replication commentary]**
- Direct replication of **Bollen, Mao & Zeng (2011)** (below). **Cannot reproduce** the result; the effect appears in BMZ's subsample but **not in the backward-extended sample — "consistent with data snooping"**; **no out-of-sample power.** Real-world coda: **Derwent Capital's "Twitter fund" (launched 2010) closed early 2012.**

**Bollen, Mao & Zeng (2011), "Twitter mood predicts the stock market," *Journal of Computational Science* 2(1), 1–8. [lower tier — CS journal]**
- 2008 (~10 months); DJIA; OpinionFinder + GPOMS moods (lagged "Calm" dimension), SOFNN model; claimed ~86.7% DJIA directional accuracy. **Treat as non-robust** per the Lachanski-Pav replication failure above.

**Sprenger et al. (2014), "Tweets and Trades," *European Financial Management* 20(5), 926–957. [A-/B]** — ~250k S&P 100 cashtag tweets (2010); Naïve Bayes; daily aggregation. Sentiment ↔ abnormal returns, volume→volume, disagreement→volatility. **Association study, not a costed strategy.**

**Ranco et al. (2015), *PLOS ONE* 10(9):e0138441. [general science]** — DJIA 30, 2013–2014, ~1.5M tweets, SVM. Only **during Twitter-volume peaks** do ~4–10-day CARs align with sentiment (~1–2% CAR); no cost/tradability analysis.

---

## Does the literature support a sentiment moving-average signal?

**Partial, and with an important sign qualification.**

**What IS supported:**
- **Smoothing/aggregating lengthens the horizon.** Heston & Sinha (2017, FAJ) is the cleanest evidence: daily news predicts 1–2 days, but **weekly-averaged** sentiment predicts a full quarter. RavenPack operationalizes exactly a **90-day SMA** of net-positive news as its factor. FEARS itself is a cross-sectional *average* of search terms. So the *mechanical idea* of smoothing a sentiment series is used and defensible — chiefly because it **cuts turnover** (helping net-of-cost performance) and isolates a slower, more persistent component.
- **Short-horizon predictability exists** from curated/higher-signal social sources (Chen et al. Seeking Alpha ~3 months; Renault intraday StockTwits; Bartov Twitter pre-earnings).

**What is NOT well supported (and contradicts the naive rule):**
- **Sign is often contrarian, not momentum.** At the aggregate/index level, high sentiment predicts *lower* future returns (Baker-Wurgler, Huang et al., Jiang et al., Da-Engelberg-Gao FEARS reversal, Tetlock intra-week reversal, Huang & Nolan on WSB). A rule that goes **long when smoothed sentiment is high** is fighting this body of evidence. A **contrarian / capitulation-buying** version (buy when sentiment is *depressed*, as in the project's Book E framing) aligns better with the literature.
- **Short-horizon Reddit-specific findings are momentum-then-reversal.** Reichenbach & Walther: short-run inflation, **significant negative alpha at 1 year**, unfavorable Sharpe. Naive long-following captures the reversal on the wrong side if held.
- **Magnitudes are typically sub-cost** (Antweiler-Frank "very small vs plausible transaction costs"; FEARS ~20 bps with a one-day data lag; Chen et al. ~2.5 bp/day gross).
- **No verified peer-reviewed paper** builds a **moving-average-of-Reddit-sentiment** signal and reports an **after-cost tradable Sharpe.** The only MA-filtered Reddit result found (an RIT MS thesis claiming ~80% accuracy filtering below the 90-day MA) is **grey literature**, not peer-reviewed, no cost controls.

**Short vs long horizon:**
- **Intraday/daily:** mostly **noise-trader / reversal** effects (Renault, FEARS, Tetlock, Antweiler-Frank). Hard to trade net of cost.
- **Weeks–quarter:** curated sentiment (Chen et al.) and **smoothed** news (Heston-Sinha) show genuine but modest drift; smoothing is what buys the horizon.
- **~1 year+:** Reddit longs **reverse** (Reichenbach-Walther).

**Bottom line:** The literature supports *smoothing a sentiment series* as a horizon-extending, turnover-reducing device, but it **does not support a naive long-when-smoothed-sentiment-is-high rule for anonymous Reddit sentiment**. The weight of evidence favors a **contrarian sign**, warns of **sub-cost magnitudes**, and documents a **post-GameStop regime break** (Bradley et al.) plus **null replications** (Kim & Kim; Lachanski-Pav) that are direct cautions for the project's experimental Book E.

---

## Signal Definition (a testable specification)

A defensible, literature-consistent version:

1. **Ingest:** Daily Reddit posts (r/wallstreetbets, r/stocks, r/investing) per ticker; require a minimum post count per ticker-day to avoid thin-data noise.
2. **Score:** Per-post sentiment via a finance-tuned classifier (VADER + Loughran-McDonald as in Reichenbach-Walther, or a fine-tuned transformer/LLM as in Huang-Nolan). Aggregate to a ticker-day **net-bullishness** index; optionally weight by post count (attention weighting).
3. **Smooth:** Compute a **moving average / EWMA** of the ticker-day sentiment index (candidate windows spanning the literature: ~5-day, 20-day, and 90-day à la RavenPack; also test weekly aggregation à la Heston-Sinha). Smoothing serves to cut turnover and isolate persistent sentiment.
4. **Signal — test BOTH signs (this is the key experiment):**
   - **Momentum arm:** long when smoothed sentiment is high / crosses above its longer MA.
   - **Contrarian arm:** long when smoothed sentiment is depressed / capitulated (favored by the aggregate-level evidence) — this matches the project's Book E capitulation framing.
5. **Cross-sectional form:** Rank universe by smoothed sentiment; long top-N (or bottom-N for contrarian), equal-weight, rebalance on a fixed cadence (weekly/monthly to keep turnover and costs sane).
6. **Costs:** Apply the project's standard 0.1%/trade + slippage; report **gross AND net** Sharpe. Compare smoothed vs un-smoothed to quantify the turnover/net-Sharpe tradeoff.
7. **Benchmarks:** vs buy-hold, vs a random-sentiment placebo, and vs a pure-attention (post-volume) signal to isolate whether *sentiment direction* adds anything beyond attention.

---

## Data Requirements

- **Reddit text:** Historical post/comment archive with timestamps and ticker extraction (cashtags + NER). Pushshift-style archives cover the meme era; note **coverage/API gaps post-2023** (Reddit API changes) — a live book faces data-availability risk.
- **Minimum history:** Per project rules, ≥2 years and earliest-available. Reddit stock-discussion volume is thin pre-2018 and dominated by the **2020–2021 meme spike** — a severe regime-concentration problem for any backtest.
- **Sentiment model:** Finance-tuned lexicon or classifier; store per-post scores to allow re-aggregation and MA-window sweeps.
- **Prices:** Aligned daily (and intraday if testing Renault-style effects) returns for the discussed universe; corporate-action-adjusted.
- **Controls:** Post volume/attention, past returns (to test the Kim-Kim "sentiment follows returns" reverse-causality), VIX, earnings dates.
- **Point-in-time discipline:** Sentiment for day *t* must use only posts available before the execution bar; respect any data-release lag (cf. FEARS one-day SVI lag).

---

## Risks & Caveats

- **Wrong sign risk (biggest):** The aggregate literature is **contrarian** (Baker-Wurgler, Huang et al., Jiang et al., FEARS, Huang-Nolan). A naive "long when sentiment high" rule may be systematically on the wrong side. **Test both signs.**
- **Regime dependence / 2021 meme era:** Reddit stock chatter is dominated by the 2020–2021 spike. Bradley et al. (2024, RFS) show WSB DD predictability was **eliminated post-GameStop** — the exact regime a live book runs in. Any backtest Sharpe is likely a 2020–2021 artifact.
- **Post-cost magnitudes:** Antweiler-Frank (sub-cost), FEARS (~20 bps + data lag), Chen et al. (~2.5 bp/day gross). Naive high-turnover sentiment signals often have **negative net Sharpe**; smoothing helps only by cutting turnover.
- **Long-run reversal:** Reichenbach-Walther find **significantly negative 1-year alpha** and unfavorable Sharpe for Reddit recommendation longs. Holding period matters enormously.
- **Null / replication failures:** Kim & Kim (2014) find **no** message-board return predictability on a large sample (sentiment follows prices). Lachanski-Pav (2017) show the famous Bollen "Twitter mood" result is **data-snooping / not out-of-sample**; the fund built on it (Derwent) **closed within ~1 year**.
- **Overfitting / data-snooping:** MA window, threshold, top-N, and holding period form a large search space over a short, regime-concentrated sample. High risk of an in-sample optimum with no OOS validity. Pre-register the grid; hold out the meme era.
- **Decay since publication:** Anomaly alphas commonly erode ~50% post-publication (McLean-Pontiff-style); social-sentiment signals are widely known and crowded, so any edge likely decays fast.
- **Data availability:** Post-2023 Reddit API restrictions threaten a reliable live feed; sentiment-model drift and platform-population changes further undermine stationarity.
- **Reality check on Book E's claimed Sharpe ~1.97:** No peer-reviewed source supports a buy-the-hype Reddit long at that Sharpe; the closest journal evidence (Reichenbach-Walther) shows the **opposite** (unfavorable Sharpe, negative long-run alpha). Treat Book E as experimental/data-limited, consistent with the project's own flagging.

---

## Sources

Message boards / media (Tier S):
- https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2004.00662.x (Antweiler-Frank JF) · https://www.sfu.ca/~kkasa/frank.pdf
- https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x (Tetlock JF) · https://business.columbia.edu/sites/default/files-efs/pubfiles/3097/Tetlock_Media_Sentiment_JF.pdf
- https://academic.oup.com/rfs/article-abstract/27/5/1367/1581938 (Chen et al. RFS) · https://www.bhwang.com/pdf/wisdom-of-crowds.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S0167268114001206 (Kim & Kim, JEBO — null)

Search / composite sentiment (Tier S):
- https://rady.ucsd.edu/faculty/directory/engelberg/pub/portfolios/FEARS.pdf · https://academic.oup.com/rfs/article-abstract/28/1/1/1682440 (FEARS)
- https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2006.00885.x (Baker-Wurgler)
- https://academic.oup.com/rfs/article-abstract/28/3/791/1576380 (Huang, Jiang, Tu, Zhou)
- https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12852 (Cookson-Niessner JF) · https://www.sciencedirect.com/science/article/abs/pii/S0304405X2400093X (Social Signal JFE — metadata only)

Twitter / StockTwits (Tier A + lower):
- https://www.sciencedirect.com/science/article/abs/pii/S0378426617301589 (Renault JBF) · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3010856
- https://publications.aaahq.org/accounting-review/article-abstract/93/3/25/4062 (Bartov et al.) · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2782236
- https://www.sciencedirect.com/science/article/abs/pii/S0378426616301595 (Sun-Najand-Shen JBF)
- https://www.sciencedirect.com/science/article/abs/pii/S187775031100007X (Bollen-Mao-Zeng) · https://arxiv.org/pdf/1010.3003
- https://econjwatch.org/articles/shy-of-the-character-limit-twitter-mood-predicts-the-stock-market-revisited (Lachanski-Pav replication)
- https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-036X.2013.12007.x (Sprenger et al.)
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0138441 (Ranco et al.)
- https://link.springer.com/article/10.1007/s42521-023-00102-z (StockTwits Classified Sentiment, Digital Finance)

News-sentiment / smoothing / practitioner:
- https://rpc.cfainstitute.org/research/financial-analysts-journal/2017/news-vs-sentiment-predicting-stock-returns-from-news-stories (Heston-Sinha FAJ) · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2792559
- https://www.ravenpack.com/research/constructing-sentiment-factor (RavenPack 90-day SMA)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2662288 (Jiang-Lee-Martin-Zhou, Manager Sentiment JFE)

Reddit / WallStreetBets / GameStop:
- https://academic.oup.com/rfs/article-abstract/37/5/1409/7486572 (Bradley et al. RFS) · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3806065 · https://russelljame.com/wsb_9_3_2021.pdf
- https://link.springer.com/article/10.1007/s42521-023-00084-y (Reichenbach-Walther, Digital Finance)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4798796 (Huang-Nolan, "Dumb Money?")
- https://onlinelibrary.wiley.com/doi/10.1111/fire.12438 (Fernandez-Perez et al., Financial Review) · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4795372
- https://www.sciencedirect.com/science/article/abs/pii/S154461232100221X (Hasso et al., Finance Research Letters)

*Verification notes: Bradley et al. exact CARs/alphas and sample end-date could not be extracted (paywalled) — no numeric magnitude cited. "Social Signal" (JFE) verified by metadata only. Net-of-cost Sharpe figures for Heston-Sinha, Tetlock, Sun-Najand-Shen, and Chen et al. were not disclosed on accessible pages and are not fabricated. No peer-reviewed paper using a moving average of Reddit sentiment as a costed trading signal was found; the closest MA-filtered Reddit result is a non-peer-reviewed MS thesis and is flagged as grey literature.*
