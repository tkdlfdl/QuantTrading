# Deploy the Paper-Trading Bot to AWS Lightsail (Windows)

A step-by-step runbook to move the `live/` engine to an always-on Windows VM so it trades
through US market hours without your laptop. **~30–45 min, ~$17–28/month.**

The bot is light, but Windows Server itself uses ~1.5 GB RAM and pandas peaks ~1 GB, so
**pick the 4 GB plan ($28/mo)** — the 2 GB plan ($17/mo) is workable but tight.

---

## 1. Create the Lightsail instance (AWS console)

1. Sign in at <https://lightsail.aws.amazon.com>
2. **Create instance**
   - Region: pick one near US markets (e.g. **us-east-1 / Virginia**)
   - Platform: **Microsoft Windows**
   - Blueprint: **Windows Server 2022**
   - Plan: **4 GB RAM / 2 vCPU / 80 GB SSD** (~$28/mo) — or 2 GB (~$17/mo) if budget-tight
   - Name it `trading-bot`, **Create instance**
3. Wait ~2–3 min until it shows **Running**.
4. (Recommended) Attach a **static IP**: Networking → Create static IP → attach to the
   instance (free while attached). Keeps the address stable across reboots.

## 2. Connect via RDP

- In Lightsail, click the instance → **Connect using RDP** (browser-based), or
- Download the RDP file / use the default Administrator password (Lightsail → instance →
  **Account page** shows it) with the Windows **Remote Desktop** app.

## 3. Install Python on the VM

1. In the VM's Edge browser, download **Python 3.12** from <https://python.org/downloads>
2. Run the installer →
   - ✅ **"Install for all users"**  (so the SYSTEM scheduled task can find it)
   - ✅ **"Add python.exe to PATH"**
   - Install
3. Verify in a Command Prompt: `python --version` → should print `Python 3.12.x`

## 4. Get the code onto the VM

**Option A — git (recommended for code):**
```
cd C:\
git clone -b mean-reversion https://github.com/tkdlfdl/QuantTrading.git Trading
```
*(the live engine is committed on the `mean-reversion` branch)*
*(install Git for Windows first if needed: <https://git-scm.com/download/win>)*

> First push the `live/` code from your laptop so it's in the repo:
> `git add live/ .gitignore requirements.txt` → `git commit -m "live paper-trading engine"`
> → `git push`.  The `.gitignore` already excludes `live/state/` so **your API secret is
> never pushed** — good.

**Option B — copy the whole folder:** zip `C:\Users\sailk\desktop\Trading` on your laptop,
upload to the VM (RDP clipboard for small files, or via an S3 bucket / Lightsail object
storage for the 188 MB `data/cache`), unzip to `C:\Trading`.

## 5. Install dependencies

```
cd C:\Trading
pip install -r requirements.txt
```

## 6. Bring over data + credentials

- **Data cache** (`data/cache/`, ~188 MB) is gitignored, so with Option A copy it across
  separately (S3 or RDP), OR let it rebuild: the first nightly run re-downloads it
  (slow, ~15–20 min, one-time). Copying is faster and deterministic.
- **Credentials:** recreate `C:\Trading\live\state\alpaca_creds.json` on the VM:
  ```json
  { "api_key": "PK...", "secret_key": "your-secret" }
  ```
  (Create the `live\state` folder first if it doesn't exist.)

## 7. Smoke-test (no orders)

```
cd C:\Trading
set PYTHONIOENCODING=utf-8
python -m live.run_intraday --once --force            # dry-run path, builds a target
python -m live.run_intraday --once --force --live     # confirms broker connects (logs orders; market gate off)
```
Check `live\state\orders.log` and that the broker line says **LIVE (paper account connected)**.
If you want the VM to be the *only* trader, **stop the laptop's tasks first** (see §10) so
the two don't both submit.

## 8. Register the scheduled tasks (on the VM, as Administrator)

Open **Command Prompt as Administrator** and run:
```
schtasks /Create /TN "PaperTradingIntraday" /TR "C:\Trading\live\run_intraday_vm.bat" /SC HOURLY /MO 1 /ST 09:35 /RU SYSTEM /RL HIGHEST /F
schtasks /Create /TN "PaperTradingDaily"   /TR "C:\Trading\live\run_daily_vm.bat"   /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:07 /RU SYSTEM /RL HIGHEST /F
```
The `*_vm.bat` wrappers use relative paths + `python` from PATH, so they work regardless of
the folder/username. SYSTEM + a 24/7 VM means it never misses a fire.

*(The VM is desktop-class — no battery — so the battery settings we fixed on the laptop are
not needed here.)*

## 9. Verify

```
schtasks /Run /TN "PaperTradingIntraday"
type live\state\intraday.log
```
During market hours you'll see a real reconcile; outside hours the gate skips. Watch fills
in the **Alpaca dashboard → Orders / Positions**.

## 10. Decommission the laptop tasks (avoid double-trading)

On your **laptop**, once the VM is live, remove its tasks so only the VM trades:
```
schtasks /Delete /TN "PaperTradingIntraday" /F
schtasks /Delete /TN "PaperTradingDaily" /F
```
(Or leave the laptop in dry-run by removing `--live` from its bats.)

---

## Notes
- **Time zone:** Lightsail Windows defaults to UTC. The bot's market gate uses
  `America/New_York` internally, so it's correct regardless of the VM clock. The schedule
  `/ST 09:35` is VM-local — since the gate is the real guard, exact local times only need to
  bracket the US session; hourly firing covers it.
- **Cost control:** stop the instance from the Lightsail console when not needed (you still
  pay for storage/static-IP, but not compute). For live trading, leave it running.
- **Security:** never commit `alpaca_creds.json`; the `.gitignore` already blocks
  `live/state/`. Rotate the Alpaca paper key if it ever leaks.
- **Single source of truth:** run live `--live` in exactly ONE place (VM) to avoid two
  bots fighting over the same paper account.
