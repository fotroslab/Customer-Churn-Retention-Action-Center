# Deployment & Public Sharing Guide (`DEPLOYMENT.md`)

This guide explains how to publish both the **GitHub Repository**, the **Free Public Interactive Web Demo (Streamlit Community Cloud)**, and the **Power BI Report**—entirely through your web browser without needing to install Git CLI or VS Code.

---

## 1. Environment Distinction

| Link / Artifact Type | Purpose | Persistence & Access |
|---|---|---|
| **Development Sandbox Preview** | Live preview server running during active development session | Temporary session URL; stops when sandbox closes |
| **Public Hosted Demo (Streamlit Cloud)** | Permanent portfolio link (`https://...streamlit.app`) for employers | Free, browser-ready, no login or Python required for visitors |
| **Power BI Desktop (`.pbix`)** | Native interactive BI file opened locally on Windows | Local file + static 4-page PDF/PNG preview in GitHub |
| **Power BI `Publish to web`** | Optional public Power BI iframe link | Requires work/school M365 Fabric tenant + admin enablement; **exposes underlying data publicly** |

---

## 2. Browser-Only GitHub Upload (No Git CLI Needed)

1. Sign in to GitHub (`https://github.com/fotroslab/`).
2. Click **New repository**:
   - **Repository name:** `customer-churn-intelligence-action-center`
   - **Visibility:** `Public` (required for free Streamlit Community Cloud hosting with unlimited viewers)
   - Leave *"Initialize with a README"* **unchecked** (the project package already includes `README.md`, `LICENSE`, and `.gitignore`).
   - Click **Create repository**.
3. Extract the clean release archive `customer-churn-intelligence-action-center-release.zip` on your computer.
4. On the empty GitHub repository page, click **uploading an existing file**, drag and drop all folders/files from inside `customer-churn-intelligence-action-center/` into the browser window, enter commit message `"Initial release: SQL + Calibrated LightGBM + Power BI Churn Action Center"`, and click **Commit changes**.
   - *(Tip: If GitHub's web uploader limits batch file counts, upload top-level files + `src/`, `sql/`, `config/`, `.streamlit/` first, then `data/`, `models/`, `powerbi/`, `scripts/`, `tests/` in a second commit—or use **GitHub Desktop** in two clicks.)*

---

## 3. Free Public Web App Deployment (Streamlit Community Cloud)

Streamlit Community Cloud hosts lightweight Python/ML portfolio apps for free and lets any recruiter or hiring manager interact with all 6 views immediately in an incognito browser tab.

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account (`fotroslab`).
2. Click **Create app** → **Deploy a public app from GitHub**:
   - **Repository:** `fotroslab/customer-churn-intelligence-action-center`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL (custom subdomain):** e.g. `churn-intelligence-action-center`
3. Click **Advanced settings**:
   - Select **Python version:** `3.11` (matches `runtime.txt`).
   - Do **not** add any secrets unless connecting to a remote database; the bundled SQLite warehouse (`data/warehouse/churn_intelligence.db`) and serialized LightGBM artifact (`models/model_bundle.joblib`) work out of the box.
4. Click **Deploy!**. First build takes ~2 minutes.
5. **Post-Deployment Verification Checklist:**
   - Copy the `https://churn-intelligence-action-center.streamlit.app` URL and open it in an **Incognito / Private browser window** (logged out of GitHub/Streamlit) to confirm recruiters can access it without signing in.
   - Note on free-tier hibernation: Streamlit Community Cloud puts idle free apps to sleep after several days of zero traffic; visitors simply click *"Yes, get this app back up"* (wakes in ~15–20 seconds).

---

## 4. Power BI Desktop Assembly & Power BI Service Sharing Options

### Option A — Portfolio Standard (100% Free, Zero Tenant Restrictions)
1. Follow `powerbi/POWER_BI_BUILD_GUIDE_FA.md` on Windows Power BI Desktop (free download) to import `data/powerbi_mart/*.csv` (or connect to SQL Server Express `ChurnIntelligenceDB`), apply `powerbi/theme/ChurnIntelligenceDarkTheme.json`, paste `powerbi/dax/measures_library.dax`, and save `powerbi/Customer_Churn_Intelligence_Action_Center.pbix`.
2. Commit the saved `.pbix` file (< 5 MB) to the `powerbi/` folder in your GitHub repository alongside `powerbi/snapshots/PowerBI_Report_Pages_Preview.pdf`.
3. Employers with Power BI Desktop can download `.pbix`, while non-Windows / executive reviewers immediately inspect the 4-page PNG/PDF snapshots in `README.md` AND use the live Streamlit app!

### Option B — Power BI Service `Publish to web (public)` (Conditional)
Per current Microsoft Fabric / Power BI documentation:
- Requires a **Microsoft work/school or M365 Developer tenant account** (consumer `@gmail.com` / `@outlook.com` addresses cannot sign up for Power BI Service).
- Publishing from **My Workspace** works with a Fabric (Free) license **only if** the tenant administrator explicitly enables the *"Publish to web"* tenant setting in the Fabric Admin Portal (shared workspaces require Power BI Pro or Premium Per User, and the link remains active only while the publisher's license/account remains active).
- **Mandatory Security Warning:** `Publish to web` makes both the visuals and underlying dataset accessible to **anyone on the internet without authentication** and does not support Row-Level Security (RLS). Use it **exclusively** with public/synthetic datasets such as this IBM Telco benchmark (`SYN-CUST-XXXXX`), never with real company or customer PII.
