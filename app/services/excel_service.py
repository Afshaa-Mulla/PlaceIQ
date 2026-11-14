import io
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from app.services.data_service import load_students, load_companies, load_roles, filter_students


def _style_worksheet(ws, header_fill_hex="1E293B"):
    header_fill = PatternFill(start_color=header_fill_hex, end_color=header_fill_hex, fill_type="solid")
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=10, color="0F172A")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws.row_dimensions[1].height = 26

    for row_idx in range(2, ws.max_row + 1):
        fill = zebra_fill if row_idx % 2 == 0 else None
        for col_idx in range(1, ws.max_column + 1):
            c = ws.cell(row=row_idx, column=col_idx)
            c.font = body_font
            c.border = thin_border
            if fill:
                c.fill = fill

    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 12
        for row_idx in range(1, min(ws.max_row + 1, 250)):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, min(38, len(str(val)) + 3))
        ws.column_dimensions[col_letter].width = max_len


def _append_df(ws, df: pd.DataFrame):
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))


def build_excel_report(report_type: str = "student", batch: str = "ALL", branch: str = "ALL", company_id: str = "") -> io.BytesIO:
    wb = Workbook()
    default_ws = wb.active

    if report_type == "batch":
        df = filter_students(batch=batch, branch=branch)
        ws_sum = default_ws
        ws_sum.title = "Batch & Branch Summary"

        grp = (
            df.groupby(["batch", "branch", "branch_name"], as_index=False)
            .agg(
                Total_Students=("student_id", "count"),
                Placed_Students=("placed", "sum"),
                Avg_CGPA=("cgpa", lambda s: round(s.mean(), 2)),
                Avg_Readiness=("readiness_score", lambda s: round(s.mean(), 1)),
                Avg_Tech_Index=("tech_index", lambda s: round(s.mean(), 1)),
                Avg_Soft_Index=("soft_index", lambda s: round(s.mean(), 1)),
                Avg_Package_LPA=("package_lpa", lambda s: round(s[s > 0].mean(), 2) if (s > 0).any() else 0.0),
                Max_Package_LPA=("package_lpa", "max"),
            )
        )
        grp["Placement_Rate_%"] = (grp["Placed_Students"] / grp["Total_Students"] * 100).round(1)
        _append_df(ws_sum, grp)
        _style_worksheet(ws_sum, "0F172A")

        # Add native bar chart on summary tab
        if len(grp) >= 2:
            chart = BarChart()
            chart.title = "Placement Rate (%) by Cohort"
            chart.y_axis.title = "Placement %"
            chart.x_axis.title = "Cohort Row"
            data_ref = Reference(ws_sum, min_col=12, min_row=1, max_row=len(grp) + 1)
            chart.add_data(data_ref, titles_from_data=True)
            chart.width = 18
            chart.height = 10
            ws_sum.add_chart(chart, f"A{len(grp) + 4}")

        ws_read = wb.create_sheet("Readiness Breakdown")
        rb = df.groupby(["branch", "readiness_band"], as_index=False).agg(Count=("student_id", "count"))
        _append_df(ws_read, rb)
        _style_worksheet(ws_read, "1E3A8A")

    elif report_type == "company":
        companies = load_companies()
        roles = load_roles()
        students = load_students()
        placed = students[students["placed"] == 1]

        ws_comp = default_ws
        ws_comp.title = "Recruiter Analytics"
        hires = placed.groupby("placed_company_id", as_index=False).agg(
            Total_Hires=("student_id", "count"),
            Actual_Avg_LPA=("package_lpa", lambda s: round(s.mean(), 2)),
            Highest_Offer_LPA=("package_lpa", "max"),
        )
        merged = companies.merge(hires, left_on="company_id", right_on="placed_company_id", how="left").fillna(
            {"Total_Hires": 0, "Actual_Avg_LPA": 0.0, "Highest_Offer_LPA": 0.0}
        )
        merged["Total_Hires"] = merged["Total_Hires"].astype(int)
        if "placed_company_id" in merged.columns:
            merged = merged.drop(columns=["placed_company_id"])
        _append_df(ws_comp, merged)
        _style_worksheet(ws_comp, "0F172A")

        ws_roles = wb.create_sheet("Job Roles & Cutoffs")
        role_cols = [
            "role_id", "company_name", "tier", "role_title", "package_band",
            "package_lpa", "min_cgpa", "max_backlogs", "preferred_branches"
        ]
        _append_df(ws_roles, roles[role_cols])
        _style_worksheet(ws_roles, "065F46")

        if company_id:
            comp_row = companies[companies["company_id"] == company_id]
            if not comp_row.empty:
                min_c = float(comp_row.iloc[0]["min_cgpa"])
                max_b = int(comp_row.iloc[0]["max_backlogs"])
                elig = students[(students["cgpa"] >= min_c) & (students["backlogs"] <= max_b)].copy()
                elig = elig.sort_values(by="readiness_score", ascending=False)
                ws_short = wb.create_sheet("Eligible Shortlist")
                cols = ["student_id", "name", "batch", "branch", "cgpa", "backlogs", "readiness_score", "tech_index", "soft_index", "placed", "placed_company"]
                _append_df(ws_short, elig[cols])
                _style_worksheet(ws_short, "4338CA")

    else:
        # Student-wise comprehensive report
        df = filter_students(batch=batch, branch=branch)
        ws_stu = default_ws
        ws_stu.title = "Student Placement Master"
        cols = [
            "student_id", "name", "gender", "batch", "branch", "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
            "python_score", "sql_score", "ml_score", "dsa_score", "web_score", "cloud_score", "core_eng_score",
            "communication_score", "aptitude_score", "teamwork_score",
            "projects_count", "internships_count", "hackathons_count", "coding_rating",
            "readiness_score", "readiness_band", "placed", "placed_company", "placed_role", "package_lpa", "package_category"
        ]
        _append_df(ws_stu, df[cols])
        _style_worksheet(ws_stu, "0F172A")

        ws_kpi = wb.create_sheet("KPI Summary")
        kpi_df = pd.DataFrame([
            {"Metric": "Total Filtered Students", "Value": int(len(df))},
            {"Metric": "Placed Students", "Value": int(df["placed"].sum())},
            {"Metric": "Placement Rate (%)", "Value": round(float(df["placed"].mean() * 100), 2) if len(df) else 0},
            {"Metric": "Average CGPA", "Value": round(float(df["cgpa"].mean()), 2) if len(df) else 0},
            {"Metric": "Average Readiness Score", "Value": round(float(df["readiness_score"].mean()), 2) if len(df) else 0},
            {"Metric": "Average Package (Placed LPA)", "Value": round(float(df[df["placed"] == 1]["package_lpa"].mean()), 2) if (df["placed"] == 1).any() else 0},
        ])
        _append_df(ws_kpi, kpi_df)
        _style_worksheet(ws_kpi, "1E3A8A")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
