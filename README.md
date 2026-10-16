# Spacey 🪐
**The right room, before the crowd arrives.** A student-first room discovery and permission-request prototype for the datathon track *The Campus Has Space. Students Don’t.*

## Features
- Student-first dark interface, original animated mascot, and a three-step room-finding mission
- Room recommendations using capacity, facilities, room type, and recurring weekday class-conflict checks
- Explainable fit score with contextual warnings; no false claim of live availability
- Direct request-recording flow addressed to demo Faculty User 1–6 identities
- My Requests tracker with pending, approved, and declined states
- Demo faculty response panel to test status updates end-to-end
- Campus Insights charts using all supplied dataset categories

## Run on Windows
Open this folder in VS Code. In the PowerShell terminal:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell activation is blocked, use Command Prompt and run `.venv\Scripts\activate.bat`.

## Dataset
Keep the supplied CSV files in `data/`: `rooms.csv`, `class_schedule.csv`, `room_requests.csv`, and `events.csv`. `room_contacts.csv` is a template for authorized verified contacts. The supplied data is synthetic and is not a live campus booking system.

## Honest prototype boundaries
- Fit score is a transparent heuristic, not a trained AI confidence score.
- A missing class overlap does not guarantee that a room is available.
- Faculty User 1–6 and `facultyuserN@gmail.com` are demo identities only; these are not verified real faculty accounts and do not receive actual email.
- Sending a request records it in `data/permission_requests.csv`. The demo faculty panel simulates a response and is not secure faculty authentication.
- Local CSV storage may not persist across cloud restarts. For a real deployment, connect authenticated faculty accounts, real email delivery, and a persistent database.

## Public repository checklist
Do not commit private request logs, personal contact details, passwords, API keys, or `.streamlit/secrets.toml`. `data/permission_requests.csv` is ignored by Git.


## Round 2: ML approval estimate

Spacey trains a reproducible scikit-learn Random Forest classifier from `data/room_requests.csv` and shows the estimated approval likelihood for pending requests in **My Requests**. The target is binary: historical `Approved` versus all other recorded outcomes. Features include request month, weekday, start hour, duration, requested capacity, projector/computer needs, and purpose. A fixed-seed, stratified 80/20 holdout split is used; Campus Insights reports accuracy, macro-F1, approved recall, and a majority-class baseline.

**Important:** the provided data is synthetic, so the percentage is a demonstration estimate—not a real faculty decision, guarantee, or room-availability prediction. Re-train and validate with authorized real campus outcomes before operational use. Install the added dependency with `pip install -r requirements.txt`.
