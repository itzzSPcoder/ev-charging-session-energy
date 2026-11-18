import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 40, letter[0] - 36, 40)
        footer_text = f"Group G171 | Problem MP298: EV Charging Session Energy | Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 26, footer_text)
        self.drawString(36, 26, "SCSE3040 MLOps Mini Project Guide")
        self.restoreState()


def build_pdf(filename="SCSE3040_MLOps_G171_MP298_Project_Handbook.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")  # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0")  # Blue
    dark_neutral = colors.HexColor("#2D3748")
    light_bg = colors.HexColor("#EDF2F7")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_neutral,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        leftIndent=12,
        spaceAfter=3,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=dark_neutral,
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2C5282"),
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("SCSE3040 MLOps Mini Project --- Group G171", title_style))
    story.append(
        Paragraph(
            "<b>Problem MP298:</b> EV Charging Session Energy Prediction (Dehradun Green Charge Network)<br/>"
            "<b>Faculty Mentor:</b> Dr. Richa Sharma &nbsp;|&nbsp; <b>Batch:</b> 29 &nbsp;|&nbsp; "
            "<b>GitHub:</b> https://github.com/itzzSPcoder/ev-charging-session-energy",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=10))

    # SECTION 1
    story.append(Paragraph("1. Project Overview --- Hum Kya Bana Rahe Hain?", h1_style))
    story.append(
        Paragraph(
            "Dehradun Green Charge Network electric vehicles (2-wheelers, 3-wheelers, electric cars) ke liye charging stations chalata hai. "
            "Jaise hi koi gaadi charger se plug hoti hai, network ko <b>usi pal</b> ye predict karna hota hai ki is poore charging session mein "
            "<b>kitni energy (kWh) consume hogi</b>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Prediction ki zaroorat kyun hai?</b><br/>"
            "1. <b>Rider Cost Transparency:</b> Driver ke mobile app mein pehle hi estimated bill/cost show ho jati hai.<br/>"
            "2. <b>Grid Load Balancing:</b> Station transformer aur local grid par aane wale load ko pehle se balance kiya ja sake.<br/>"
            "<b>ML Task:</b> Ye ek <b>Regression Problem</b> hai jiska target <code>energy_kwh</code> hai, aur iska evaluation metric <b>MAE (Mean Absolute Error)</b> hai.",
            body_style,
        )
    )

    # Input Features Table
    features_data = [
        [Paragraph("Feature", table_header), Paragraph("Type", table_header), Paragraph("Meaning & Role", table_header)],
        [Paragraph("battery_start_pct", table_cell), Paragraph("Numeric (%)", table_cell), Paragraph("Battery percentage when plugged in (e.g. 20%)", table_cell)],
        [Paragraph("session_minutes", table_cell), Paragraph("Numeric (min)", table_cell), Paragraph("Planned charging duration. Rule: Must be >= 5 mins", table_cell)],
        [Paragraph("charger_kw", table_cell), Paragraph("Numeric (kW)", table_cell), Paragraph("Charger output rating (e.g. 22 kW, 50 kW)", table_cell)],
        [Paragraph("temp_c", table_cell), Paragraph("Numeric (°C)", table_cell), Paragraph("Air temperature. Rule: Must be <= 75°C", table_cell)],
        [Paragraph("vehicle_age_yr", table_cell), Paragraph("Numeric (yr)", table_cell), Paragraph("Vehicle age in years (battery degradation factor)", table_cell)],
        [Paragraph("queue_before", table_cell), Paragraph("Numeric (count)", table_cell), Paragraph("Vehicles waiting in line before this session", table_cell)],
        [Paragraph("vehicle", table_cell), Paragraph("Categorical", table_cell), Paragraph("Vehicle class: car, two_wheeler, three_wheeler", table_cell)],
        [Paragraph("station_type", table_cell), Paragraph("Categorical", table_cell), Paragraph("Location environment: Mall, Highway, Residential", table_cell)],
    ]
    t_feat = Table(features_data, colWidths=[120, 90, 330])
    t_feat.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(t_feat)
    story.append(Spacer(1, 8))

    # SECTION 2
    story.append(Paragraph("2. End-to-End MLOps Pipeline --- Kese Banaya?", h1_style))
    story.append(
        Paragraph(
            "Humne sirf ek ML model nahi banaya, balki poora enterprise MLOps production pipeline build kiya hai across 4 stages:",
            body_style,
        )
    )

    # 4 Stages Table
    stages_data = [
        [Paragraph("Stage & Timeline", table_header), Paragraph("Key Deliverables & Technology", table_header), Paragraph("Outcome / Verification", table_header)],
        [
            Paragraph("<b>Stage 1</b><br/>Data Lead", table_cell),
            Paragraph("DVC data versioning, raw CSV isolation, 7 data problem hunters in <code>src/ev_charging/data.py</code>", table_cell),
            Paragraph("DVC tracked <code>train.csv.dvc</code> & <code>later.csv.dvc</code>; 4 data problems identified.", table_cell),
        ],
        [
            Paragraph("<b>Stage 2</b><br/>Model Lead", table_cell),
            Paragraph("Data cleaning pipeline, MLflow SQLite tracking (<code>mlflow.db</code>), SHA-256 hash logging, 3 candidate models.", table_cell),
            Paragraph("4 runs tracked, best v1 Gradient Boosting model registered as Version 1.", table_cell),
        ],
        [
            Paragraph("<b>Stage 3</b><br/>Ops & Service Lead", table_cell),
            Paragraph("FastAPI service (<code>app/main.py</code>) with Pydantic validation rules, JSON logging, Docker, K8s, GitHub CI.", table_cell),
            Paragraph("Dockerfile created, K8s deployment (4 replicas) & service, 11 tests green on GitHub Actions.", table_cell),
        ],
        [
            Paragraph("<b>Stage 4</b><br/>Full Team", table_cell),
            Paragraph("PSI drift computation on <code>later.csv</code>, slice MAE, v2 retraining on train + 70% later, holdout predictions.", table_cell),
            Paragraph("v2 improved error by 57.28% (>=15%), promoted to <code>@champion</code>; 250 holdout rows predicted.", table_cell),
        ],
    ]
    t_stages = Table(stages_data, colWidths=[90, 250, 200])
    t_stages.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(t_stages)
    story.append(Spacer(1, 8))

    # Data Issues Detail Callout
    issues_box = [
        [
            Paragraph(
                "<b>The 7 Data Issues Discovered in <code>train.csv</code>:</b><br/>"
                "• <b>Leaking Column (Present):</b> <code>kwh_billed_in_app</code> ($r = 0.998$ correlation with target). Dropped from features.<br/>"
                "• <b>Duplicate Rows (Present):</b> Exactly <b>40</b> duplicate rows found and removed.<br/>"
                "• <b>Stuck Sensor (Present):</b> <code>battery_start_pct</code> stuck at 47% from <b>2026-03-13</b> to <b>2026-04-05</b> (133 rows). Handled in cleaning.<br/>"
                "• <b>Spelling Variants (Present):</b> <code>vehicle</code> had 20 spelling variants (' thr.', 'TWO_WHEELER', ' car '), cleaned to <b>3 true levels</b>.<br/>"
                "• <b>Absent Issues:</b> Unit Mixup (False), Sentinel Value (False), Target Typos (False).",
                callout_style,
            )
        ]
    ]
    t_issues = Table(issues_box, colWidths=[540])
    t_issues.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3182CE")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(t_issues)
    story.append(Spacer(1, 10))

    # SECTION 3
    story.append(Paragraph("3. How to Run & Test --- Isko Test Kese Karna Hai?", h1_style))
    story.append(
        Paragraph(
            "Humne ek <b>FastAPI REST Backend Service</b> banaya hai jo IoT chargers aur mobile apps ke sath communicate karta hai. "
            "Ise test karne ke 3 direct methods hain:",
            body_style,
        )
    )

    test_data = [
        [Paragraph("Method", table_header), Paragraph("Command / URL", table_header), Paragraph("Expected Behavior & Validation", table_header)],
        [
            Paragraph("<b>1. Swagger Web UI</b><br/>(Interactive Browser)", table_cell),
            Paragraph("Terminal: <code>uvicorn app.main:app --port 8000</code><br/>Browser: <b>http://localhost:8000/docs</b>", table_cell),
            Paragraph("Visual UI opens. Click <code>POST /predict</code> -> 'Try it out' -> enter JSON -> 'Execute'. Live prediction returned in < 0.05s.", table_cell),
        ],
        [
            Paragraph("<b>2. Automated Tests</b><br/>(Pytest Suite)", table_cell),
            Paragraph("Terminal: <code>pytest -v</code>", table_cell),
            Paragraph("Runs 11 automated unit & API tests (health check, valid prediction, rule rejects, data cleaning, PSI drift). All pass 100%.", table_cell),
        ],
        [
            Paragraph("<b>3. Terminal cURL</b><br/>(Direct HTTP)", table_cell),
            Paragraph("<code>curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{...}'</code>", table_cell),
            Paragraph("Returns JSON: <code>{\"prediction\": 28.45}</code>. Stdout logs single-line structured JSON without PII.", table_cell),
        ],
        [
            Paragraph("<b>4. Business Rule Checks</b><br/>(Validation Errors)", table_cell),
            Paragraph("Set <code>temp_c: 80</code> OR <code>session_minutes: 3</code>", table_cell),
            Paragraph("FastAPI rejects bad input with <b>HTTP 422 Unprocessable Entity</b>, preventing faulty inputs from reaching the ML model.", table_cell),
        ],
    ]
    t_test = Table(test_data, colWidths=[110, 210, 220])
    t_test.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(t_test)
    story.append(Spacer(1, 10))

    # SECTION 4
    story.append(Paragraph("4. Individual Member Roles & Form Claims --- Kiska Kya Role Hai?", h1_style))
    story.append(
        Paragraph(
            "Rubric rule ke mutabiq har member ko Bennett University ke official submission form mein apna individual contribution "
            "bharte waqt ye exact details daalni hain:",
            body_style,
        )
    )

    roles_data = [
        [Paragraph("Member & Enrolment", table_header), Paragraph("Role & Responsibilities", table_header), Paragraph("Files Authored & Owned", table_header), Paragraph("Hours", table_header)],
        [
            Paragraph("<b>Sparsh Bansal</b><br/>(S24CSEU0560)<br/><i>Ops & Service Lead</i>", table_cell),
            Paragraph("FastAPI prediction service, client validation rules (Pydantic), JSON logging, Docker containerization, Kubernetes manifests (Deployment & Service), GitHub Actions CI workflow, repo coordinator.", table_cell),
            Paragraph("<code>app/main.py</code><br/><code>tests/test_api.py</code><br/><code>Dockerfile</code>, <code>.dockerignore</code><br/><code>k8s/deployment.yaml</code><br/><code>k8s/service.yaml</code><br/><code>.github/workflows/ci.yml</code><br/><code>requirements.txt</code>, <code>README.md</code>", table_cell),
            Paragraph("<b>17 hrs</b>", table_cell),
        ],
        [
            Paragraph("<b>Siddharth Rawat</b><br/>(S24CSEU0542)<br/><i>Data Lead</i>", table_cell),
            Paragraph("DVC setup and dataset version control, detecting the 7 data issues in train.csv (leaking column, 40 duplicates, stuck sensor, vehicle spellings), data cleaning pipeline implementation.", table_cell),
            Paragraph("<code>.dvc/config</code><br/><code>data/raw/train.csv.dvc</code><br/><code>data/raw/later.csv.dvc</code><br/><code>data/raw/holdout_features.csv</code><br/><code>src/ev_charging/data.py</code><br/><code>tests/test_data_cleaning.py</code>", table_cell),
            Paragraph("<b>15 hrs</b>", table_cell),
        ],
        [
            Paragraph("<b>Tanisha Rathore</b><br/>(S24CSEU0591)<br/><i>Model Lead</i>", table_cell),
            Paragraph("MLflow tracking & SQLite registry setup, logging data_sha256 hash, candidate training (Ridge, RF, GB), PSI drift computation on temp_c, v2 model retraining & promotion rule, holdout predictions.", table_cell),
            Paragraph("<code>params.yaml</code><br/><code>src/ev_charging/features.py</code><br/><code>src/ev_charging/metrics.py</code><br/><code>src/ev_charging/models.py</code><br/><code>src/ev_charging/train.py</code><br/><code>src/ev_charging/evaluate.py</code><br/><code>src/ev_charging/predict.py</code><br/><code>mlflow.db</code>, <code>models/model.joblib</code><br/><code>predictions/holdout_predictions.csv</code><br/><code>findings.yaml</code>", table_cell),
            Paragraph("<b>16 hrs</b>", table_cell),
        ],
    ]
    t_roles = Table(roles_data, colWidths=[110, 180, 200, 50])
    t_roles.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_roles)
    story.append(Spacer(1, 10))

    # SECTION 5
    story.append(Paragraph("5. Viva Cheat-Sheet --- Most Important Questions & Answers", h1_style))
    viva_qas = [
        ("Q1. kwh_billed_in_app data leakage kyun hai?",
         "Kyunki billing column charging session khatam hone ke baad bhara jata hai. Real-world scenario mein jab gaadi plug hoti hai, ye value exist hi nahi karti. Iska correlation target ke sath 0.998 tha, isliye isko training features se hatana mandatory tha."),
        ("Q2. Stuck sensor kya tha aur kaise detect hua?",
         "Dates ko sort karke jab daily unique values dekhi gayi, toh pata chala ki 13 March 2026 se lekar 05 April 2026 tak har record mein battery_start_pct = 47% tha (133 rows). Sensor freeze ho chuka tha. Humne cleaning pipeline mein in corrupted rows ko filter kiya."),
        ("Q3. later.csv mein kaun sa feature drift hua aur kyun?",
         "temp_c sabse zyada drift hua (PSI = 3.6203). Training data January-June (winter & spring) ka tha jahan average temperature 25.2°C tha, jabki later.csv July-August (peak summer) ka tha jahan average temperature 44.0°C tha."),
        ("Q4. Model v2 ko kyun promote kiya gaya?",
         "Client ka rule tha ki retrained model (v2) ko tabhi promote karna hai jab check split (later.csv ke aakhri 30%) par uska MAE v1 se kam se kam 15% better ho. v1 ka MAE 5.86 tha aur v2 ka 2.50 tha (57.28% improvement), isliye v2 ko @champion alias assign kiya gaya."),
        ("Q5. Client ke 2 validation rules kahan implement kiye gaye hain?",
         "Pydantic schema (PredictRequest) mein Field(..., le=75.0) for temp_c and Field(..., ge=5) for session_minutes ke zariye. Agar koi rule violate hota hai toh request seedha HTTP 422 reject ho jati hai aur model tak nahi pahunchti."),
        ("Q6. Kubernetes deployment ki specifications kya hain?",
         "Deployment name: mp298-api, 4 replicas, container port 8000, readiness probe (initialDelay: 5s) aur liveness probe (initialDelay: 20s). Service name: mp298-svc, cluster port 80 forwarding to container port 8000."),
    ]

    for q, a in viva_qas:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, bullet_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {filename}")


if __name__ == "__main__":
    build_pdf()
