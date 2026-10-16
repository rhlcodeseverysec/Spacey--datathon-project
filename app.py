from pathlib import Path
from datetime import date, datetime, time, timedelta
import base64
import html
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
REQUEST_LOG = DATA / "permission_requests.csv"
CONTACTS_PATH = DATA / "room_contacts.csv"

st.set_page_config(page_title="Spacey · Find your space", page_icon="🪐", layout="wide", initial_sidebar_state="collapsed")

# ---------- visual identity ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
:root { color-scheme: dark; }
.stApp { background-color:#020308; background-image: radial-gradient(1px 1px at 20px 30px, #fff 98%, transparent), radial-gradient(1px 1px at 90px 80px, #9ecbff 98%, transparent), radial-gradient(2px 2px at 160px 40px, #fff 98%, transparent), radial-gradient(1px 1px at 230px 120px, #c4b5fd 98%, transparent), radial-gradient(1px 1px at 300px 70px, #fff 98%, transparent), radial-gradient(ellipse at 50% 0%, #0b1022 0%, #03040a 55%, #000 100%); background-size: 340px 180px, 270px 210px, 390px 250px, 310px 220px, 360px 240px, 100% 100%; background-attachment:fixed; color:#f4f7ff; font-family:'Nunito',sans-serif; }
html, body, [class*="css"] { font-family:'Nunito',sans-serif; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.2rem; padding-bottom:2.5rem; max-width:1180px; }
h1,h2,h3 { letter-spacing:-.035em; font-weight:900!important; color:#f7f9ff; }
p, label, .stMarkdown { color:#dce5fa; }
[data-testid="stMetric"] { background:#17233b; border:1px solid #2b3d5e; padding:15px; border-radius:18px; }
.stButton>button,.stFormSubmitButton>button { border-radius:15px!important; border:1px solid #5de8b0!important; background:#5de8b0!important; color:#102033!important; font-weight:900!important; min-height:2.8rem; transition:transform .16s ease,filter .16s ease; }
.stButton>button:hover,.stFormSubmitButton>button:hover { transform:translateY(-2px); filter:brightness(1.06); }
.stButton>button[kind="secondary"] { background:#1a2944!important;color:#eaf1ff!important;border-color:#334c70!important; }
.stTextInput input,.stNumberInput input,.stDateInput input,.stTimeInput input,.stTextArea textarea,[data-baseweb="select"]>div { background:#131f36!important; border-radius:12px!important; border-color:#334767!important; }
[data-testid="stRadio"] > div { gap:8px; }
[data-testid="stRadio"] label { background:#15233b; border:1px solid #2a3c5e; border-radius:14px; padding:8px 14px; }
[data-testid="stRadio"] label:has(input:checked) { background:#23453f; border-color:#5de8b0; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#2b3d5e!important; border-radius:20px!important; background:rgba(19,30,51,.7); }
.hero { background:linear-gradient(118deg,#142039,#122b35 70%,#163b34); border:1px solid #365a6b; border-radius:28px; padding:24px 28px; margin:8px 0 18px; overflow:hidden; }
.kicker { font-size:.76rem; font-weight:900; letter-spacing:.18em; color:#65f2b7; }
.hero h1 { font-size:clamp(2.1rem,4.6vw,3.6rem); margin:.2rem 0 .35rem; }
.hero p { font-size:1.05rem; max-width:700px; color:#d0def5; }
.mascot { animation:floaty 3s ease-in-out infinite; filter:drop-shadow(0 10px 25px #5de8b02c); }
@keyframes floaty { 0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-9px) rotate(2deg)} }
@keyframes scan { 0%{transform:translateX(-110%)}100%{transform:translateX(110%)} }
@keyframes popin { from{opacity:0;transform:translateY(12px) scale(.985)}to{opacity:1;transform:translateY(0) scale(1)} }
.pill { display:inline-block;padding:5px 11px;border-radius:999px;background:#153c37;color:#75f4bd;border:1px solid #286b59;font-size:.8rem;font-weight:900; }
.subtle { color:#9fb1d1; font-size:.91rem; }
.room-card { animation:popin .38s ease both; }
.scanline { height:5px;border-radius:99px;background:#20344c;overflow:hidden;margin:12px 0; }
.scanline:after { content:'';display:block;height:100%;width:42%;background:linear-gradient(90deg,transparent,#5de8b0,transparent);animation:scan 1.7s ease-in-out infinite; }
.stProgress > div > div { background:#5de8b0; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(filename):
    p = DATA / filename
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

rooms = load_csv("rooms.csv")
schedule = load_csv("class_schedule.csv")
historical_requests = load_csv("room_requests.csv")
events = load_csv("events.csv")
contacts = load_csv("room_contacts.csv")

for col in ["capacity", "projector", "computers", "accessible", "power_outlets", "floor"]:
    if col in rooms.columns: rooms[col] = pd.to_numeric(rooms[col], errors="coerce").fillna(0)
for col in ["start_hour", "duration_hours"]:
    if col in schedule.columns: schedule[col] = pd.to_numeric(schedule[col], errors="coerce")

if rooms.empty or schedule.empty:
    st.error("Spacey couldn't load the room data. Make sure rooms.csv and class_schedule.csv are inside the data folder.")
    st.stop()

# ---------- state ----------
def init(key, value):
    if key not in st.session_state: st.session_state[key] = value

init("page", "Home")
init("nav_widget", "Home")
init("mission_step", 0)
init("space_points", 0)
init("completed_missions", [])
init("purpose", "Club meeting")
init("chosen_room", "")
init("results_ready", False)
init("search_date_value", date.today())
init("search_time_value", time(14, 0))
init("search_duration_value", 1.0)
init("search_attendees_value", 30)
init("search_projector_value", False)
init("search_computers_value", False)
init("search_accessible_value", False)
init("search_outlets_value", 0)
init("search_type_value", "Any")
init("request_form_open", False)
init("request_sent", False)

NAV = ["Home", "Find a Room", "My Requests", "Campus Insights"]


def mascot(width=104):
    p = BASE / "assets" / "spacey-mascot.svg"
    if not p.exists(): return "🪐"
    svg = p.read_text(encoding="utf-8")
    return f'<img class="mascot" style="width:{width}px;height:{width}px" alt="Spacey mascot" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}" />'


def go(page):
    st.session_state.page = page
    st.rerun()


def load_student_requests():
    if not REQUEST_LOG.exists(): return pd.DataFrame()
    try: return pd.read_csv(REQUEST_LOG, dtype=str).fillna("")
    except Exception: return pd.DataFrame()


def save_student_requests(df):
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(REQUEST_LOG, index=False)


def faculty_directory():
    return [{"name": f"Faculty User {i}", "email": f"facultyuser{i}@gmail.com", "demo": True} for i in range(1, 7)]


# Round 2 ML: estimate historical approval likelihood, not room availability.
APPROVAL_FEATURES = ["month", "weekday", "start_hour", "duration_hours", "required_capacity",
                     "projector_required", "computers_required", "purpose"]
NUMERIC_FEATURES = ["month", "start_hour", "duration_hours", "required_capacity",
                    "projector_required", "computers_required"]
CATEGORICAL_FEATURES = ["weekday", "purpose"]


def prepare_approval_features(frame):
    x = pd.DataFrame(index=frame.index)
    dates = pd.to_datetime(frame.get("request_date", pd.Series(index=frame.index, dtype=str)), errors="coerce")
    x["month"] = dates.dt.month
    x["weekday"] = dates.dt.day_name().fillna("Unknown")
    for col in ["start_hour", "duration_hours", "required_capacity", "projector_required", "computers_required"]:
        x[col] = pd.to_numeric(frame.get(col, pd.Series(index=frame.index, dtype=float)), errors="coerce")
    x["purpose"] = frame.get("purpose", pd.Series(index=frame.index, dtype=str)).fillna("Unknown").astype(str)
    return x[APPROVAL_FEATURES]


@st.cache_resource
def train_approval_model(csv_path, file_mtime):
    """Train a reproducible binary classifier and evaluate it on held-out rows."""
    data = pd.read_csv(csv_path)
    needed = {"request_date", "start_hour", "duration_hours", "required_capacity",
              "projector_required", "computers_required", "purpose", "status"}
    if not needed.issubset(data.columns):
        return None, {"error": "Historical request data is missing required columns."}
    data = data[data["status"].notna()].copy()
    data["approved_target"] = data["status"].astype(str).str.strip().str.lower().eq("approved").astype(int)
    if len(data) < 30 or data["approved_target"].nunique() < 2:
        return None, {"error": "Not enough historical outcome variety to train a model."}
    x = prepare_approval_features(data)
    y = data["approved_target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("features", prep),
        ("classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=8,
                                               class_weight=None, random_state=42,
                                               n_jobs=-1)),
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    majority = int(y_train.mean() >= 0.5)
    baseline_pred = [majority] * len(y_test)
    metrics = {
        "rows": int(len(data)),
        "approved_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "approved_recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "baseline_macro_f1": float(f1_score(y_test, baseline_pred, average="macro", zero_division=0)),
        "test_rows": int(len(y_test)),
    }
    return model, metrics


def approval_probability(request_row):
    """Return model probability for an incoming request and evaluation metadata."""
    data_path = DATA / "room_requests.csv"
    try:
        model, metrics = train_approval_model(str(data_path), data_path.stat().st_mtime)
        if model is None:
            return None, metrics
        one = pd.DataFrame([{
            "request_date": request_row.get("requested_date", ""),
            "start_hour": int(str(request_row.get("start_time", "00:00")).split(":")[0]),
            "duration_hours": request_row.get("duration_hours", 1),
            "required_capacity": request_row.get("expected_attendance", 1),
            "projector_required": request_row.get("projector_required", 0),
            "computers_required": request_row.get("computers_required", 0),
            "purpose": request_row.get("purpose", "Unknown"),
        }])
        features = prepare_approval_features(one)
        probability = float(model.predict_proba(features)[0][list(model.named_steps["classifier"].classes_).index(1)])
        return probability, metrics
    except Exception as exc:
        return None, {"error": f"Model unavailable: {exc}"}


def verified_contact_for(room_id):
    if contacts.empty or "room_id" not in contacts.columns: return None
    sub = contacts[contacts.room_id.astype(str) == str(room_id)].copy()
    needed = {"verified", "source", "last_verified"}
    if sub.empty or not needed.issubset(sub.columns): return None
    yes = sub["verified"].astype(str).str.strip().str.lower().isin(["yes", "true", "1", "verified"])
    source = sub["source"].fillna("").astype(str).str.strip().ne("")
    day = sub["last_verified"].fillna("").astype(str).str.strip().ne("")
    sub = sub[yes & source & day]
    return sub.iloc[0].to_dict() if not sub.empty else None


def calculate_matches():
    d = st.session_state.search_date_value
    start = st.session_state.search_time_value.hour + st.session_state.search_time_value.minute / 60
    duration = float(st.session_state.search_duration_value)
    end = start + duration
    attendees = int(st.session_state.search_attendees_value)
    day = d.strftime("%a")
    sc = schedule.copy()
    sc["weekday_norm"] = sc["weekday"].astype(str).str[:3].str.title()
    sc = sc[sc.weekday_norm == day]
    sc["class_start"] = pd.to_numeric(sc["start_hour"], errors="coerce")
    sc["class_end"] = sc.class_start + pd.to_numeric(sc["duration_hours"], errors="coerce")
    overlap = sc[(sc.class_start < end) & (sc.class_end > start)]
    conflict_ids = set(overlap.room_id.astype(str))
    found = rooms[rooms.capacity >= attendees].copy()
    if st.session_state.search_projector_value: found = found[found.projector == 1]
    if st.session_state.search_computers_value: found = found[found.computers == 1]
    if st.session_state.search_accessible_value: found = found[found.accessible == 1]
    outlets = int(st.session_state.search_outlets_value)
    if outlets: found = found[found.power_outlets >= outlets]
    room_type = st.session_state.search_type_value
    if room_type != "Any": found = found[found.room_type.astype(str) == room_type]
    found["recorded_class_conflict"] = found.room_id.astype(str).isin(conflict_ids)
    found["capacity_surplus"] = found.capacity - attendees
    found["fit_score"] = found.apply(lambda r: max(0, min(100, round(100 - min(float(r.capacity_surplus), 200)*0.15 - (25 if r.recorded_class_conflict else 0)))), axis=1)
    return found.sort_values(["recorded_class_conflict", "capacity_surplus"], ascending=[True, True]), start, end

# ---------- header and navigation ----------
st.markdown(f'''<div class="hero"><div style="display:flex;align-items:center;gap:18px">{mascot(112)}<div><div class="kicker">YOUR CAMPUS. YOUR SPACE.</div><h1>Spacey</h1><p><b>The right room, before the crowd arrives.</b><br/>Find a space for your squad, then send a permission request without the campus scavenger hunt.</p><span class="pill">✦ Find it · Fit it · Make it happen</span></div></div></div>''', unsafe_allow_html=True)
st.session_state.nav_widget = st.session_state.page
nav = st.radio("Main navigation", NAV, horizontal=True, key="nav_widget", label_visibility="collapsed")
st.session_state.page = nav

# small stats are purposeful and student-facing, not a dashboard
req_df = load_student_requests()
if nav == "Home":
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("## Your next mission")
        st.markdown("### Find your squad a room! 🧑‍🚀")
        st.write("Study session, club meeting, or project sprint? Tell Spacey what you need and get a shortlist with clear reasons for every match.")
        purpose = st.radio("What's the mission?", ["Study session", "Club meeting", "Project work", "Event / workshop"], horizontal=True, index=["Study session", "Club meeting", "Project work", "Event / workshop"].index(st.session_state.purpose) if st.session_state.purpose in ["Study session", "Club meeting", "Project work", "Event / workshop"] else 1, key="home_purpose")
        st.session_state.purpose = purpose
        if st.button("🚀 Find a room", use_container_width=True):
            st.session_state.mission_step = 1
            st.session_state.results_ready = False
            st.session_state.space_points += 5
            st.session_state.completed_missions = list(dict.fromkeys(st.session_state.completed_missions + ["started"]))
            go("Find a Room")
    with right:
        st.markdown(mascot(156), unsafe_allow_html=True)
        st.markdown("### Spacey says…")
        st.info("“Your squad brings the ideas. I'll help you find the space.”")
        st.progress(min(100, len(st.session_state.completed_missions) * 25), text=f"Mission progress · {min(4, len(st.session_state.completed_missions))} of 4")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("🪐 Space points", st.session_state.space_points)
    with p2:
        st.metric("🏫 Spaces to explore", len(rooms))
    with p3:
        st.metric("📨 Your requests", len(req_df))
    st.markdown("### Your mission map")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🔎 Find your fit")
            st.write("Match your group size, facilities and time.")
            if st.button("Explore rooms →", key="home_explore", use_container_width=True): go("Find a Room")
    with c2:
        with st.container(border=True):
            st.markdown("#### ✉️ Ask faculty")
            st.write("Send a request and track the response in one place.")
            if st.button("My requests →", key="home_requests", use_container_width=True): go("My Requests")
    with c3:
        with st.container(border=True):
            st.markdown("#### 🌌 Explore campus")
            st.write("Discover patterns in the historical campus data.")
            if st.button("Campus insights →", key="home_insights", use_container_width=True): go("Campus Insights")

elif nav == "Find a Room":
    step = int(st.session_state.mission_step)
    st.markdown("## 🧭 Find your space")
    labels = ["1 · Tell us what you need", "2 · Check your mission", "Explore your matches"]
    st.progress(min(1.0, max(0.05, (step + 1) / 3)), text=labels[min(step, 2)])
    if step <= 0:
        st.markdown("### Step 1 — Build your room mission")
        with st.container(border=True):
            a, b, c = st.columns(3)
            with a: st.date_input("When?", key="search_date_value")
            with b: st.time_input("Start time", step=timedelta(minutes=30), key="search_time_value")
            with c: st.selectbox("How long?", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], format_func=lambda x: f"{x:g} hour" + ("s" if x != 1 else ""), key="search_duration_value")
            a, b, c, d = st.columns(4)
            with a: st.number_input("People in your squad", 1, 500, key="search_attendees_value")
            with b: st.checkbox("Need a projector", key="search_projector_value")
            with c: st.checkbox("Need computers", key="search_computers_value")
            with d: st.checkbox("Step-free access", key="search_accessible_value")
            a, b = st.columns(2)
            with a: st.number_input("Minimum power outlets", 0, 100, key="search_outlets_value")
            with b: st.selectbox("Room vibe", ["Any"] + sorted(rooms.room_type.dropna().astype(str).unique().tolist()), key="search_type_value")
        if st.button("Next: review my mission →", use_container_width=True):
            st.session_state.mission_step = 1
            st.rerun()
    elif step == 1:
        st.markdown("### Step 2 — Quick mission check")
        d = st.session_state.search_date_value
        t = st.session_state.search_time_value
        dur = float(st.session_state.search_duration_value)
        end_dt = datetime.combine(d, t) + timedelta(hours=dur)
        with st.container(border=True):
            st.markdown(f"#### 🗓️ {d:%A, %d %B %Y}")
            st.markdown(f"**Time:** {t:%H:%M}–{end_dt:%H:%M}  ·  **Squad:** {int(st.session_state.search_attendees_value)} people")
            st.markdown(f"**Mission:** {html.escape(st.session_state.purpose)}")
            facilities = []
            if st.session_state.search_projector_value: facilities.append("Projector")
            if st.session_state.search_computers_value: facilities.append("Computers")
            if st.session_state.search_accessible_value: facilities.append("Step-free access")
            if int(st.session_state.search_outlets_value): facilities.append(f"{int(st.session_state.search_outlets_value)}+ power outlets")
            st.markdown("**Must-haves:** " + (", ".join(facilities) if facilities else "No extra facilities"))
            st.markdown("**Room type:** " + st.session_state.search_type_value)
        back, search = st.columns(2)
        with back:
            if st.button("← Change choices", use_container_width=True):
                st.session_state.mission_step = 0
                st.rerun()
        with search:
            if st.button("✨ Find my space", use_container_width=True):
                st.session_state.mission_step = 2
                st.session_state.results_ready = True
                st.session_state.space_points += 5
                st.session_state.completed_missions = list(dict.fromkeys(st.session_state.completed_missions + ["searched"]))
                st.rerun()
    else:
        if st.session_state.request_form_open:
            st.markdown("## ✉️ Draft your room request")
            st.caption("Review your selected room and send a request to a demo faculty account. Demo Gmail addresses are placeholders and do not receive real email.")
            chosen_room_id = str(st.session_state.chosen_room)
            selected = rooms[rooms.room_id.astype(str) == chosen_room_id]
            if not selected.empty:
                room_row = selected.iloc[0]
                st.markdown(f"### 🏫 {html.escape(chosen_room_id)} · {html.escape(str(room_row.get('room_type', 'Room')))}")
                st.caption(f"{html.escape(str(room_row.get('building', 'Unknown building')))} · {int(room_row.get('capacity', 0))} seats")
            if st.button("← Go back and change room", key="back_change_room", use_container_width=True):
                st.session_state.request_form_open = False
                st.rerun()
            with st.form("send_request_form", clear_on_submit=False):
                faculty = st.selectbox("Send to faculty", faculty_directory(), format_func=lambda f: f"{f['name']} · {f['email']}")
                col1, col2 = st.columns(2)
                with col1: student_name = st.text_input("Your name")
                with col2: student_email = st.text_input("Your college email (optional)")
                purpose = st.text_input("What is the room for?", value=st.session_state.purpose)
                col1, col2, col3 = st.columns(3)
                with col1: req_date = st.date_input("Requested date", value=st.session_state.search_date_value, key="send_date")
                with col2: req_time = st.time_input("Start time", value=st.session_state.search_time_value, key="send_time")
                with col3: req_duration = st.selectbox("Duration", [0.5,1.0,1.5,2.0,3.0,4.0], index=[0.5,1.0,1.5,2.0,3.0,4.0].index(float(st.session_state.search_duration_value)), format_func=lambda x:f"{x:g} hours", key="send_duration")
                expected = st.number_input("Expected attendance", 1, 500, int(st.session_state.search_attendees_value), key="send_attendance")
                notes = st.text_area("Message to faculty (optional)", placeholder="Add context for your request")
                submitted = st.form_submit_button("🚀 Send request", use_container_width=True)
            if submitted:
                if not student_name.strip() or not purpose.strip():
                    st.error("Please add your name and the purpose of the request.")
                else:
                    now = datetime.now()
                    old = load_student_requests()
                    chosen_profile = selected.iloc[0] if not selected.empty else pd.Series(dtype=object)
                    record = {"request_id": f"SP-{now:%Y%m%d%H%M%S}-{len(old)+1:03d}", "created_at": now.isoformat(timespec="seconds"), "room_id": chosen_room_id, "requested_date": req_date.isoformat(), "start_time": req_time.strftime("%H:%M"), "duration_hours": str(req_duration), "expected_attendance": str(expected), "projector_required": str(int(pd.to_numeric(chosen_profile.get("projector", 0), errors="coerce") or 0)), "computers_required": str(int(pd.to_numeric(chosen_profile.get("computers", 0), errors="coerce") or 0)), "purpose": purpose.strip(), "requester_name": student_name.strip(), "requester_email": student_email.strip(), "faculty_name": faculty["name"], "faculty_email": faculty["email"], "faculty_is_demo": "yes", "notes": notes.strip(), "status": "Pending faculty response", "response_note": ""}
                    save_student_requests(pd.concat([old, pd.DataFrame([record])], ignore_index=True))
                    st.session_state.space_points += 10
                    st.session_state.completed_missions = list(dict.fromkeys(st.session_state.completed_missions + ["request"]))
                    st.session_state.request_form_open = False
                    st.session_state.request_sent = True
                    st.toast("Request added to My Requests · +10 Space points")
                    st.balloons()
                    st.rerun()
        else:
            st.markdown("### Scanning your campus for spaces")
            st.markdown(f'''<div style="display:flex;align-items:center;gap:16px;padding:12px 0">{mascot(90)}<div style="flex:1"><b>Spacey is checking the room map</b><div class="subtle">Matching capacity · checking recorded class times · comparing facilities</div><div class="scanline"></div></div></div>''', unsafe_allow_html=True)
            matches, start, end = calculate_matches()
            dt = st.session_state.search_date_value
            end_dt = datetime.combine(dt, st.session_state.search_time_value) + timedelta(hours=float(st.session_state.search_duration_value))
            st.caption(f"{dt:%A, %d %B} · {st.session_state.search_time_value:%H:%M}–{end_dt:%H:%M} · {int(st.session_state.search_attendees_value)} people · {st.session_state.purpose}")
            st.markdown(f"## {len(matches)} room{'s' if len(matches) != 1 else ''} to explore")
            st.info("Schedule check: Spacey compares recurring weekday class records. No recorded overlap is not a live availability confirmation; faculty approval is still needed.")
            if matches.empty:
                st.warning("No rooms match every choice yet. Try a smaller group or remove one optional facility.")
                st.markdown(mascot(95), unsafe_allow_html=True)
                st.write("Spacey says: “Every great mission has a plan B. Let's adjust your choices!”")
                if st.button("← Change my choices", use_container_width=True):
                    st.session_state.mission_step = 0
                    st.rerun()
            else:
                clean = matches[~matches.recorded_class_conflict]
                if len(clean): st.success(f"{len(clean)} match(es) meet the physical requirements with no recorded class overlap.")
                else: st.warning("All matching rooms have recorded class overlaps. You can still ask faculty, but don't assume these rooms are free.")
                for i, (_, r) in enumerate(matches.head(12).iterrows()):
                    rid = str(r.room_id)
                    conflict = bool(r.recorded_class_conflict)
                    contact = verified_contact_for(rid)
                    with st.container(border=True):
                        left, right = st.columns([4, 1])
                        with left:
                            st.markdown(f"#### 🏫 {rid} · {html.escape(str(r.get('room_type', 'Room')))}")
                            st.markdown(f"**{html.escape(str(r.get('building','Unknown building')))}** · Floor {int(r.get('floor',0))} · **{int(r.capacity)} seats**")
                            st.write(f"{'📽️ Projector available' if r.projector else 'No projector'}  ·  {'💻 Computers available' if r.computers else 'No computers'}  ·  {'♿ Step-free access' if r.accessible else 'Accessibility not marked'}  ·  🔌 {int(r.power_outlets)} outlets")
                            why = [f"Capacity fit: fits {int(st.session_state.search_attendees_value)} people with {int(r.capacity_surplus)} spare seats", ("Projector available" if r.projector else "Projector unavailable")]
                            if st.session_state.search_computers_value:
                                why.append("Computers available" if r.computers else "Computers unavailable")
                            if st.session_state.search_accessible_value:
                                why.append("Step-free access available" if r.accessible else "Step-free access not marked")
                            if int(st.session_state.search_outlets_value):
                                why.append(f"Power outlets: {int(r.power_outlets)} available")
                            why.append("Recorded class clash found at this time" if conflict else "No recorded class clash at this time")
                            st.caption("Why this match: " + " · ".join(why))
                            if conflict: st.warning("A class is recorded during this time. Ask before planning around it.")
                            else: st.success("No overlapping class record found for this weekday/time — this is not a live booking confirmation.")
                            if contact:
                                st.caption(f"Verified contact: {contact.get('contact_name','Contact')} · {contact.get('role','Authorized contact')}")
                            else:
                                st.caption("Faculty contacts for specific rooms aren't verified in the dataset; choose a demo faculty user when sending a request.")
                        with right:
                            st.metric("Fit", f"{int(r.fit_score)}%")
                            st.caption("Room-fit score")
                            if st.button("Request this room →", key=f"request_{rid}", use_container_width=True):
                                st.session_state.chosen_room = rid
                                st.session_state.request_form_open = True
                                st.session_state.space_points += 5
                                st.rerun()
            if st.button("← Change mission choices", key="change_after_results"):
                st.session_state.mission_step = 0
                st.session_state.results_ready = False
                st.rerun()
        if st.session_state.request_sent:
            st.success("🚀 Request sent successfully! Your mission is now waiting for the faculty response.")
            if st.button("View my request →", key="view_my_request", use_container_width=True):
                st.session_state.request_sent = False
                go("My Requests")

elif nav == "My Requests":
    st.markdown("## 🎒 My Requests")
    st.write("Every request has a home here. Check who you asked, when you asked, and whether the request is still waiting for a response.")
    req_df = load_student_requests()
    if req_df.empty:
        st.info("Your request backpack is empty. Find a room and send your first faculty request!")
        if st.button("Find a room →", use_container_width=True): go("Find a Room")
    else:
        counts = req_df.status.value_counts().to_dict() if "status" in req_df.columns else {}
        c1,c2,c3 = st.columns(3)
        c1.metric("⏳ Pending", int(counts.get("Pending faculty response",0)))
        c2.metric("✅ Approved", int(counts.get("Approved",0)))
        c3.metric("↩️ Declined", int(counts.get("Declined",0)))
        for _, r in req_df.iloc[::-1].iterrows():
            status = str(r.get("status", "Pending faculty response"))
            symbol = "⏳" if status == "Pending faculty response" else ("✅" if status == "Approved" else ("↩️" if status == "Declined" else "📌"))
            with st.container(border=True):
                col1, col2 = st.columns([3,1])
                with col1:
                    st.markdown(f"#### {symbol} {html.escape(str(r.get('room_id','Room')))} · {html.escape(str(r.get('purpose','Room request')))}")
                    st.markdown(f"**Faculty:** {html.escape(str(r.get('faculty_name','Faculty')))} · `{html.escape(str(r.get('faculty_email','')) )}`")
                    st.caption(f"Requested {r.get('requested_date','')} · {r.get('start_time','')} · {r.get('duration_hours','')} hour(s) · {r.get('expected_attendance','')} people")
                    st.caption(f"Request ID: {r.get('request_id','')} · Sent: {r.get('created_at','')}")
                    if status == "Pending faculty response":
                        probability, model_info = approval_probability(r.to_dict())
                        if probability is not None:
                            st.markdown("**🤖 Estimated approval chance**")
                            st.progress(probability)
                            st.markdown(f"### {probability:.0%}")
                            st.caption("ML estimate from historical request patterns; this is not a faculty decision or a guarantee. The supplied training data is synthetic.")
                        else:
                            st.caption("Approval estimate unavailable: " + str(model_info.get("error", "model could not be trained")))
                    if str(r.get("response_note", "")).strip(): st.info("Faculty note: " + str(r.get("response_note")))
                with col2:
                    if status == "Approved": st.success("Approved")
                    elif status == "Declined": st.error("Declined")
                    else: st.warning("Pending")
        st.caption("Requests are stored in this app's local CSV for the prototype. A public cloud demo may not preserve records after a restart unless connected to a persistent database.")

elif nav == "Campus Insights":
    st.markdown("## 🌌 Campus Insights")
    st.write("A curious look at the supplied historical data — what spaces exist, what students have requested, and where planning might improve.")
    st.caption("These charts describe the synthetic datathon dataset, not live campus occupancy or confirmed room availability.")
    a,b,c = st.columns(3)
    a.metric("🏫 Room profiles", len(rooms))
    b.metric("📚 Class schedule records", len(schedule))
    c.metric("🎟️ Historical events", len(events))
    x,y = st.columns(2)
    with x:
        st.markdown("### 🪑 Average capacity by room type")
        if "room_type" in rooms.columns:
            st.bar_chart(rooms.groupby("room_type").capacity.mean().sort_values(ascending=False))
    with y:
        st.markdown("### 📬 Historical request outcomes")
        if not historical_requests.empty and "status" in historical_requests.columns:
            st.bar_chart(historical_requests.status.fillna("Unknown").value_counts())
        else: st.info("Historical request outcomes aren't available in the supplied file.")
    with st.expander("🤖 Round 2 ML · approval model evaluation"):
        model_path = DATA / "room_requests.csv"
        model, ml_metrics = train_approval_model(str(model_path), model_path.stat().st_mtime) if model_path.exists() else (None, {"error": "Historical request data not found."})
        if model is None:
            st.warning(ml_metrics.get("error", "Model could not be trained."))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Historical requests", f"{ml_metrics['rows']:,}")
            m2.metric("Test macro-F1", f"{ml_metrics['macro_f1']:.3f}")
            m3.metric("Test accuracy", f"{ml_metrics['accuracy']:.1%}")
            st.write(f"Approved recall: **{ml_metrics['approved_recall']:.1%}** · Majority-baseline accuracy: **{ml_metrics['baseline_accuracy']:.1%}** · Majority-baseline macro-F1: **{ml_metrics['baseline_macro_f1']:.3f}** · Test records: **{ml_metrics['test_rows']}**")
            st.caption("Binary target: Approved vs all other historical statuses. Random 80/20 stratified split with fixed seed. Because the supplied data is synthetic and contains no guaranteed room allocation, these estimates are demonstration signals—not real-campus probabilities or room availability predictions.")
    if not events.empty and {"expected_attendance", "actual_attendance"}.issubset(events.columns):
        st.markdown("### 🎟️ Expected vs actual event attendance")
        chart = events[["expected_attendance", "actual_attendance"]].apply(pd.to_numeric, errors="coerce").dropna().head(100)
        if not chart.empty: st.line_chart(chart.reset_index(drop=True))
    st.markdown("### 💡 What this can help campus teams explore")
    st.write("- Which room types tend to offer more seats.\n- How historical requests are distributed across outcomes.\n- Whether event attendance regularly differs from expectations.")
    st.caption(f"Dataset snapshot: {len(rooms)} rooms · {len(schedule)} class records · {len(historical_requests)} historical requests · {len(events)} events")

# Contextual limitations only, not a large homepage disclaimer
with st.expander("About Spacey · data and demo notes"):
    st.write("Spacey uses the synthetic educational dataset supplied for the datathon. Room-fit scores are transparent heuristics, and schedule checks use recurring weekday class records. No overlap does not prove live availability; room permission must still be approved by an authorized campus representative.")
    st.write("Faculty User 1–6 and their Gmail-style addresses are demo identities only. Sending a request records it in this prototype; it does not deliver a real email. Faculty approval and decline actions are not available in the student interface.")
    st.write("Requests are stored locally in data/permission_requests.csv. Do not enter sensitive personal information into a public demo.")
st.markdown(f'<div style="display:flex;align-items:center;gap:9px;opacity:.85;margin-top:24px">{mascot(38)}<span class="subtle">Spacey · The right room, before the crowd arrives.</span></div>', unsafe_allow_html=True)
