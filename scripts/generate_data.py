"""
Synthetic Dataset Generator for PlaceIQ
Generates realistic multi-branch engineering & data science placement records (~1,800 students across 4 batches),
45 companies, and 60 job roles with hidden causal rules + stochastic noise.
NOTE: 100% synthetic dataset created for academic & analytics demonstration.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

BRANCHES = [
    ("CSE", "Computer Science & Engineering", 0.24),
    ("IT", "Information Technology", 0.19),
    ("DS", "Data Science & AI", 0.17),
    ("ECE", "Electronics & Communication", 0.15),
    ("EE", "Electrical Engineering", 0.09),
    ("MECH", "Mechanical Engineering", 0.09),
    ("CIVIL", "Civil Engineering", 0.07),
]

BATCHES = [2023, 2024, 2025, 2026]

FIRST_NAMES = [
    "Aarav", "Ananya", "Rohan", "Priya", "Vikram", "Sneha", "Karthik", "Divya", "Siddharth", "Meera",
    "Aditya", "Neha", "Arjun", "Kavya", "Rahul", "Pooja", "Nikhil", "Shreya", "Varun", "Isha",
    "Pranav", "Riya", "Abhishek", "Tanvi", "Harsh", "Aishwarya", "Manish", "Sakshi", "Kunal", "Aditi",
    "Yash", "Nandini", "Tushar", "Swati", "Akash", "Komal", "Saurabh", "Ankita", "Deepak", "Bhavna",
    "Omkar", "Pallavi", "Shubham", "Gauri", "Tejas", "Mrunal", "Chinmay", "Vaishnavi", "Sahil", "Sonal",
    "Devansh", "Kritika", "Rishabh", "Simran", "Mayank", "Jhanvi", "Aniket", "Nidhi", "Gaurav", "Mitali"
]

LAST_NAMES = [
    "Sharma", "Patel", "Verma", "Nair", "Kulkarni", "Reddy", "Iyer", "Gupta", "Joshi", "Deshmukh",
    "Mehta", "Singh", "Rao", "Das", "Choudhury", "Pillai", "Bhat", "Menon", "Jain", "Agarwal",
    "Naik", "Gokhale", "More", "Pawar", "Shinde", "Sawant", "Chavan", "Kadam", "Jadhav", "Patil",
    "Banerjee", "Mukherjee", "Bose", "Sen", "Malhotra", "Kapoor", "Khanna", "Bansal", "Mittal", "Tiwari"
]

TECH_SKILLS = [
    "python_score", "sql_score", "ml_score", "dsa_score",
    "web_score", "cloud_score", "core_eng_score"
]

SOFT_SKILLS = [
    "communication_score", "aptitude_score", "teamwork_score"
]

COMPANIES_SPEC = [
    # (company_id, name, tier, domain, min_cgpa, max_backlogs, preferred_branches, base_lpa, roles)
    ("CMP01", "Aetheris AI Labs", "Product / AI", "Artificial Intelligence", 8.2, 0, ["CSE", "DS", "IT"], 18.5, [
        ("ML Research Engineer", "Premium", 21.0, {"python_score": 82, "ml_score": 85, "dsa_score": 78, "sql_score": 70}),
        ("Applied Data Scientist", "Premium", 17.5, {"python_score": 80, "ml_score": 82, "sql_score": 76, "communication_score": 74}),
    ]),
    ("CMP02", "Vanguard Cloud Systems", "Product / Cloud", "Cloud Infrastructure", 8.0, 0, ["CSE", "IT", "ECE"], 16.0, [
        ("Cloud Platform Engineer", "Premium", 16.8, {"cloud_score": 82, "python_score": 76, "dsa_score": 75, "web_score": 70}),
        ("Site Reliability Engineer", "High", 14.5, {"cloud_score": 80, "python_score": 74, "sql_score": 68, "teamwork_score": 72}),
    ]),
    ("CMP03", "QuantEdge FinTech", "FinTech", "Algorithmic Finance", 8.3, 0, ["CSE", "DS", "IT", "ECE"], 19.0, [
        ("Quantitative Analyst", "Premium", 20.5, {"python_score": 84, "ml_score": 80, "aptitude_score": 85, "dsa_score": 80}),
        ("Backend FinTech Developer", "Premium", 17.2, {"dsa_score": 84, "sql_score": 78, "python_score": 78, "cloud_score": 70}),
    ]),
    ("CMP04", "NexusCore Software", "Product", "Enterprise SaaS", 7.8, 0, ["CSE", "IT", "DS"], 14.0, [
        ("Full Stack Product Engineer", "High", 14.2, {"web_score": 82, "dsa_score": 76, "sql_score": 74, "cloud_score": 68}),
        ("Software Development Engineer I", "High", 13.8, {"dsa_score": 82, "python_score": 75, "web_score": 72, "teamwork_score": 72}),
    ]),
    ("CMP05", "Stratos DataWorks", "Analytics", "Data Engineering", 7.5, 1, ["DS", "CSE", "IT", "ECE"], 12.2, [
        ("Data Engineer", "High", 12.8, {"sql_score": 84, "python_score": 78, "cloud_score": 75, "dsa_score": 68}),
        ("BI & Analytics Specialist", "High", 11.2, {"sql_score": 82, "python_score": 72, "communication_score": 78, "aptitude_score": 74}),
    ]),
    ("CMP06", "Kinetix Robotics", "Core / DeepTech", "Autonomous Systems", 7.8, 0, ["ECE", "MECH", "CSE", "EE"], 13.5, [
        ("Embedded & Control Engineer", "High", 13.5, {"core_eng_score": 84, "python_score": 72, "dsa_score": 68, "teamwork_score": 74}),
        ("Perception & Vision Engineer", "High", 14.2, {"ml_score": 80, "python_score": 80, "core_eng_score": 72, "dsa_score": 72}),
    ]),
    ("CMP07", "Helios Semicon", "Core / Hardware", "VLSI & Semiconductors", 7.9, 0, ["ECE", "EE"], 14.8, [
        ("RTL Design Engineer", "High", 15.0, {"core_eng_score": 86, "aptitude_score": 78, "python_score": 65, "teamwork_score": 72}),
    ]),
    ("CMP08", "Titan Heavy Engineering", "Core Manufacturing", "Industrial Systems", 7.2, 1, ["MECH", "EE", "CIVIL"], 9.2, [
        ("Graduate Engineer Trainee - Design", "Mid", 9.4, {"core_eng_score": 82, "aptitude_score": 72, "teamwork_score": 75, "communication_score": 70}),
        ("Supply Chain & Ops Analyst", "Mid", 8.8, {"sql_score": 68, "aptitude_score": 76, "core_eng_score": 74, "communication_score": 74}),
    ]),
    ("CMP09", "Civitas InfraGroup", "Core Infrastructure", "Smart Cities & Civil", 7.0, 1, ["CIVIL", "MECH", "EE"], 8.5, [
        ("Structural & BIM Engineer", "Mid", 8.6, {"core_eng_score": 82, "teamwork_score": 74, "aptitude_score": 70, "communication_score": 70}),
    ]),
    ("CMP10", "VoltGrid Power", "Core Energy", "Smart Grid & EV", 7.3, 1, ["EE", "ECE", "MECH"], 9.8, [
        ("Power Systems & EV Engineer", "Mid", 9.8, {"core_eng_score": 82, "python_score": 64, "aptitude_score": 74, "teamwork_score": 72}),
    ]),
    ("CMP11", "Accenture Strategy & Tech", "Consulting", "Digital Transformation", 6.8, 1, ["CSE", "IT", "DS", "ECE", "EE", "MECH", "CIVIL"], 7.8, [
        ("Associate Business Analyst", "Mid", 8.2, {"communication_score": 80, "sql_score": 70, "aptitude_score": 75, "teamwork_score": 76}),
        ("Packaged App Development Analyst", "Mid", 7.2, {"python_score": 66, "web_score": 66, "sql_score": 66, "communication_score": 72}),
    ]),
    ("CMP12", "Deloitte Consulting", "Consulting", "Risk & Analytics", 7.2, 0, ["CSE", "IT", "DS", "ECE", "EE"], 9.5, [
        ("Analytics & AI Consultant", "Mid", 10.2, {"sql_score": 78, "python_score": 74, "communication_score": 82, "ml_score": 70}),
        ("Technology Risk Analyst", "Mid", 8.8, {"cloud_score": 68, "communication_score": 80, "aptitude_score": 76, "sql_score": 68}),
    ]),
    ("CMP13", "Infosys Specialist Programmer", "IT Services+", "Digital Engineering", 7.5, 0, ["CSE", "IT", "DS", "ECE"], 9.5, [
        ("Specialist Programmer", "Mid", 9.5, {"dsa_score": 80, "python_score": 76, "sql_score": 72, "aptitude_score": 75}),
    ]),
    ("CMP14", "TCS Digital & Prime", "IT Services+", "Enterprise Solutions", 7.0, 1, ["CSE", "IT", "DS", "ECE", "EE", "MECH"], 7.5, [
        ("TCS Prime Engineer", "Mid", 9.0, {"dsa_score": 76, "python_score": 74, "cloud_score": 68, "communication_score": 72}),
        ("TCS Digital Developer", "Mid", 7.0, {"python_score": 68, "web_score": 66, "sql_score": 66, "aptitude_score": 70}),
    ]),
    ("CMP15", "Cognizant GenC Next", "IT Services", "Application Services", 6.5, 1, ["CSE", "IT", "DS", "ECE", "EE", "MECH", "CIVIL"], 6.2, [
        ("GenC Next Programmer", "Low", 6.5, {"python_score": 65, "sql_score": 64, "aptitude_score": 68, "communication_score": 68}),
        ("GenC Engineer", "Low", 4.8, {"aptitude_score": 62, "communication_score": 64, "teamwork_score": 65}),
    ]),
    ("CMP16", "Wipro Turbo", "IT Services", "Cloud & Infra", 6.5, 1, ["CSE", "IT", "ECE", "EE", "MECH"], 6.0, [
        ("Turbo Project Engineer", "Low", 6.2, {"web_score": 64, "sql_score": 62, "aptitude_score": 66, "communication_score": 66}),
    ]),
    ("CMP17", "Capgemini Exceller", "IT Services", "Engineering & R&D", 6.5, 1, ["CSE", "IT", "DS", "ECE", "EE", "MECH", "CIVIL"], 5.8, [
        ("Senior Analyst - Software", "Low", 5.8, {"python_score": 62, "sql_score": 60, "communication_score": 68, "teamwork_score": 68}),
    ]),
    ("CMP18", "Zomato / Blinkit Tech", "Consumer Tech", "Hyperlocal Logistics", 7.6, 0, ["CSE", "IT", "DS"], 15.5, [
        ("SDE-1 Backend & Scale", "High", 15.5, {"dsa_score": 82, "sql_score": 78, "cloud_score": 74, "python_score": 76}),
    ]),
    ("CMP19", "Razorpay Engineering", "FinTech", "Payments Infrastructure", 7.9, 0, ["CSE", "IT", "DS"], 17.5, [
        ("Software Engineer - Payments", "Premium", 17.8, {"dsa_score": 84, "web_score": 78, "sql_score": 78, "cloud_score": 74}),
    ]),
    ("CMP20", "Fractal Analytics", "Analytics", "Decision Sciences", 7.4, 0, ["DS", "CSE", "IT", "ECE"], 11.5, [
        ("Imagineer - Data Science", "High", 11.5, {"ml_score": 78, "python_score": 76, "sql_score": 76, "communication_score": 76}),
    ]),
    ("CMP21", "Mu Sigma", "Analytics", "Decision Support", 6.8, 1, ["CSE", "IT", "DS", "ECE", "EE", "MECH", "CIVIL"], 6.8, [
        ("Trainee Decision Scientist", "Low", 6.8, {"aptitude_score": 76, "sql_score": 68, "communication_score": 76, "python_score": 62}),
    ]),
    ("CMP22", "L&T Technology Services", "Engineering R&D", "Digital Manufacturing", 7.0, 1, ["MECH", "EE", "ECE", "CIVIL", "CSE"], 7.2, [
        ("Embedded & IoT Engineer", "Mid", 7.4, {"core_eng_score": 76, "python_score": 66, "teamwork_score": 72, "aptitude_score": 70}),
        ("CAD / Simulation Engineer", "Mid", 7.0, {"core_eng_score": 78, "aptitude_score": 70, "teamwork_score": 70}),
    ]),
    ("CMP23", "Siemens Digital Industries", "Core / Automation", "Industrial Automation", 7.6, 0, ["EE", "ECE", "MECH", "CSE"], 11.8, [
        ("Automation & IIoT Engineer", "High", 11.8, {"core_eng_score": 80, "python_score": 72, "cloud_score": 68, "teamwork_score": 74}),
    ]),
    ("CMP24", "Bosch Global Software", "Mobility Tech", "Automotive Software", 7.2, 0, ["ECE", "EE", "CSE", "IT", "MECH"], 8.6, [
        ("Automotive Embedded Developer", "Mid", 8.6, {"core_eng_score": 76, "python_score": 70, "dsa_score": 68, "teamwork_score": 72}),
    ]),
    ("CMP25", "ZS Associates", "Consulting", "Healthcare Analytics", 7.4, 0, ["DS", "CSE", "IT", "ECE", "EE", "MECH"], 11.2, [
        ("Decision Analytics Associate", "High", 11.4, {"sql_score": 78, "aptitude_score": 80, "communication_score": 80, "python_score": 70}),
    ]),
    ("CMP26", "JPMorgan Chase Tech", "BFSI Tech", "Global Banking Tech", 8.0, 0, ["CSE", "IT", "DS", "ECE"], 16.5, [
        ("Software Engineer - CIB", "Premium", 16.5, {"dsa_score": 80, "python_score": 78, "sql_score": 76, "communication_score": 76}),
    ]),
    ("CMP27", "Barclays Global Service", "BFSI Tech", "FinTech Operations", 7.5, 0, ["CSE", "IT", "DS", "ECE"], 12.5, [
        ("BA3 Technology Analyst", "High", 12.5, {"python_score": 74, "sql_score": 74, "cloud_score": 70, "communication_score": 75}),
    ]),
    ("CMP28", "Oracle Cloud Infra", "Product / Cloud", "Database & Cloud", 7.8, 0, ["CSE", "IT", "DS"], 15.2, [
        ("Member of Technical Staff", "High", 15.2, {"dsa_score": 82, "sql_score": 82, "cloud_score": 76, "python_score": 74}),
    ]),
    ("CMP29", "Cisco Systems", "Networking", "Enterprise Security", 7.7, 0, ["CSE", "IT", "ECE", "EE"], 14.6, [
        ("Network & Security Software Eng", "High", 14.6, {"python_score": 76, "cloud_score": 76, "dsa_score": 76, "core_eng_score": 72}),
    ]),
    ("CMP30", "Schneider Electric", "Core Energy", "Energy Automation", 7.2, 1, ["EE", "ECE", "MECH"], 8.8, [
        ("Graduate Engineer - Smart Grid", "Mid", 8.8, {"core_eng_score": 78, "aptitude_score": 72, "communication_score": 72, "teamwork_score": 74}),
    ]),
    ("CMP31", "Tata Motors EV", "Automotive", "Electric Mobility", 7.3, 1, ["MECH", "EE", "ECE"], 9.2, [
        ("EV Powertrain & Battery Eng", "Mid", 9.2, {"core_eng_score": 80, "python_score": 64, "teamwork_score": 74, "aptitude_score": 72}),
    ]),
    ("CMP32", "Jacobs Engineering", "Core Infrastructure", "Design & Consulting", 7.1, 1, ["CIVIL", "MECH", "EE"], 7.8, [
        ("Graduate Civil / Structural Eng", "Mid", 7.8, {"core_eng_score": 78, "communication_score": 72, "teamwork_score": 74, "aptitude_score": 70}),
    ]),
    ("CMP33", "Afcons Infrastructure", "Core Construction", "Heavy Civil", 6.8, 1, ["CIVIL", "MECH"], 6.8, [
        ("Site & Planning Engineer", "Low", 6.8, {"core_eng_score": 75, "teamwork_score": 75, "aptitude_score": 68}),
    ]),
    ("CMP34", "Freshworks", "Product", "SaaS CRM", 7.6, 0, ["CSE", "IT", "DS"], 13.2, [
        ("Product Software Engineer", "High", 13.2, {"web_score": 80, "dsa_score": 76, "sql_score": 72, "communication_score": 74}),
    ]),
    ("CMP35", "Zoho Corporation", "Product", "Business Suite", 7.0, 1, ["CSE", "IT", "DS", "ECE"], 8.4, [
        ("Member Technical Staff", "Mid", 8.4, {"dsa_score": 76, "web_score": 74, "python_score": 70, "aptitude_score": 74}),
    ]),
    ("CMP36", "Tiger Analytics", "Analytics", "AI Consulting", 7.4, 0, ["DS", "CSE", "IT", "ECE"], 11.0, [
        ("Analyst - Data Science", "High", 11.0, {"python_score": 76, "ml_score": 76, "sql_score": 76, "communication_score": 74}),
    ]),
    ("CMP37", "PwC India Tech", "Consulting", "Data & Cloud Advisory", 7.0, 1, ["CSE", "IT", "DS", "ECE", "EE"], 8.5, [
        ("Associate - Data & Analytics", "Mid", 8.5, {"sql_score": 74, "python_score": 68, "communication_score": 78, "teamwork_score": 75}),
    ]),
    ("CMP38", "KPMG Digital", "Consulting", "Tech Assurance", 6.9, 1, ["CSE", "IT", "DS", "ECE", "MECH"], 7.6, [
        ("Analyst - Digital Lighthouse", "Mid", 7.6, {"sql_score": 70, "communication_score": 76, "aptitude_score": 72, "teamwork_score": 74}),
    ]),
    ("CMP39", "Mindtree / LTIMindtree", "IT Services", "Cloud & Data", 6.6, 1, ["CSE", "IT", "DS", "ECE", "EE", "MECH", "CIVIL"], 5.5, [
        ("Graduate Engineer Trainee", "Low", 5.5, {"python_score": 60, "sql_score": 60, "aptitude_score": 64, "communication_score": 65}),
    ]),
    ("CMP40", "Tech Mahindra", "IT Services", "Telecom & IT", 6.2, 2, ["CSE", "IT", "ECE", "EE", "MECH", "CIVIL"], 4.6, [
        ("Associate Software Engineer", "Low", 4.6, {"aptitude_score": 60, "communication_score": 62, "teamwork_score": 64}),
    ]),
    ("CMP41", "HCLTech", "IT Services", "Digital Foundation", 6.4, 2, ["CSE", "IT", "DS", "ECE", "EE", "MECH", "CIVIL"], 4.8, [
        ("Graduate Engineer Trainee", "Low", 4.8, {"aptitude_score": 62, "communication_score": 62, "sql_score": 58}),
    ]),
    ("CMP42", "Persistent Systems", "Product Engineering", "Outsourced Product Dev", 7.2, 0, ["CSE", "IT", "DS", "ECE"], 7.8, [
        ("Software Engineer", "Mid", 7.8, {"dsa_score": 72, "python_score": 70, "web_score": 68, "sql_score": 68}),
    ]),
    ("CMP43", "PubMatic", "AdTech", "Programmatic Cloud", 7.6, 0, ["CSE", "IT", "DS"], 12.6, [
        ("Big Data & Backend Engineer", "High", 12.6, {"dsa_score": 78, "sql_score": 78, "cloud_score": 72, "python_score": 74}),
    ]),
    ("CMP44", "Godrej & Boyce", "Core Manufacturing", "Precision Engineering", 7.0, 1, ["MECH", "EE", "CIVIL"], 7.4, [
        ("Post Graduate / Graduate Trainee", "Mid", 7.4, {"core_eng_score": 76, "teamwork_score": 74, "communication_score": 72, "aptitude_score": 70}),
    ]),
    ("CMP45", "NVIDIA Graphics India", "DeepTech / AI", "GPU & AI Systems", 8.5, 0, ["CSE", "ECE", "DS"], 22.0, [
        ("System & AI Software Engineer", "Premium", 22.0, {"python_score": 86, "dsa_score": 85, "ml_score": 84, "core_eng_score": 76}),
    ]),
]


def clip(val, lo, hi):
    return float(np.clip(val, lo, hi))


def generate_all(seed=42, n_students=1800):
    rng = np.random.default_rng(seed)

    # 1. Build companies and roles tables
    companies_rows = []
    roles_rows = []
    role_counter = 1
    for cid, cname, tier, domain, min_cgpa, max_bl, pref_branches, base_lpa, rlist in COMPANIES_SPEC:
        companies_rows.append({
            "company_id": cid,
            "company_name": cname,
            "tier": tier,
            "domain": domain,
            "min_cgpa": min_cgpa,
            "max_backlogs": max_bl,
            "preferred_branches": ",".join(pref_branches),
            "avg_package_lpa": base_lpa,
            "open_roles_count": len(rlist),
        })
        for rtitle, pkg_band, pkg_lpa, req_skills in rlist:
            rid = f"ROL{role_counter:02d}"
            role_counter += 1
            row = {
                "role_id": rid,
                "company_id": cid,
                "company_name": cname,
                "tier": tier,
                "domain": domain,
                "role_title": rtitle,
                "package_band": pkg_band,
                "package_lpa": pkg_lpa,
                "min_cgpa": min_cgpa,
                "max_backlogs": max_bl,
                "preferred_branches": ",".join(pref_branches),
                "required_skills_json": json.dumps(req_skills),
            }
            for s in TECH_SKILLS + SOFT_SKILLS:
                row[f"req_{s}"] = req_skills.get(s, 0)
            roles_rows.append(row)

    df_companies = pd.DataFrame(companies_rows)
    df_roles = pd.DataFrame(roles_rows)

    # 2. Generate 1,800 students across 4 batches and all 7 branches
    branch_codes = [b[0] for b in BRANCHES]
    branch_names = {b[0]: b[1] for b in BRANCHES}
    branch_probs = [b[2] for b in BRANCHES]

    batch_probs = [0.23, 0.24, 0.26, 0.27]  # 2023..2026
    batch_market_shift = {2023: -0.03, 2024: -0.01, 2025: 0.02, 2026: 0.04}

    students = []
    for i in range(1, n_students + 1):
        sid = f"STU{i:04d}"
        batch = int(rng.choice(BATCHES, p=batch_probs))
        branch = str(rng.choice(branch_codes, p=branch_probs))
        fname = str(rng.choice(FIRST_NAMES))
        lname = str(rng.choice(LAST_NAMES))
        full_name = f"{fname} {lname}"
        gender = str(rng.choice(["Female", "Male"], p=[0.44, 0.56]))

        # Latent traits: academic diligence, coding aptitude, communication ease, core inclination
        diligence = rng.normal(0.0, 1.0)
        coding_latent = 0.45 * diligence + rng.normal(0.0, 0.9)
        soft_latent = 0.35 * diligence + rng.normal(0.0, 0.9)

        # Branch priors
        is_cs_it = branch in ("CSE", "IT")
        is_ds = branch == "DS"
        is_circuital = branch in ("ECE", "EE")
        is_core = branch in ("MECH", "CIVIL")

        # Academics
        tenth_pct = clip(79.0 + 6.5 * diligence + rng.normal(0, 4.5), 55.0, 98.5)
        twelfth_pct = clip(77.0 + 7.0 * diligence + rng.normal(0, 5.0), 54.0, 98.0)
        cgpa = round(clip(7.55 + 0.78 * diligence + rng.normal(0, 0.42), 5.2, 9.85), 2)

        # Backlogs correlated inversely with diligence
        bl_lambda = max(0.04, 0.55 - 0.35 * diligence)
        backlogs = int(np.clip(rng.poisson(bl_lambda), 0, 5))
        if cgpa >= 8.2 and rng.random() < 0.85:
            backlogs = 0

        # Technical skills (0 - 100)
        cs_boost = 14 if is_cs_it else (11 if is_ds else (2 if is_circuital else -7))
        ds_boost = 18 if is_ds else (8 if is_cs_it else (-2 if is_circuital else -8))
        web_boost = 14 if is_cs_it else (7 if is_ds else (-4 if is_circuital else -10))
        core_boost = 22 if is_core else (18 if is_circuital else -16)

        python_score = round(clip(64 + 10.5 * coding_latent + cs_boost * 0.7 + rng.normal(0, 7.5), 20, 99), 1)
        sql_score = round(clip(62 + 9.5 * coding_latent + ds_boost * 0.65 + rng.normal(0, 8.0), 18, 99), 1)
        ml_score = round(clip(54 + 10.0 * coding_latent + ds_boost * 0.95 + rng.normal(0, 8.5), 15, 99), 1)
        dsa_score = round(clip(60 + 11.5 * coding_latent + cs_boost * 0.85 + rng.normal(0, 8.5), 15, 99), 1)
        web_score = round(clip(58 + 9.0 * coding_latent + web_boost * 0.85 + rng.normal(0, 8.5), 15, 99), 1)
        cloud_score = round(clip(53 + 8.5 * coding_latent + cs_boost * 0.55 + rng.normal(0, 8.5), 15, 98), 1)
        core_eng_score = round(clip(60 + 9.5 * diligence + core_boost * 0.85 + rng.normal(0, 8.0), 18, 99), 1)

        # Soft skills
        communication_score = round(clip(66 + 10.5 * soft_latent + rng.normal(0, 7.5), 28, 98), 1)
        aptitude_score = round(clip(67 + 9.0 * diligence + 3.5 * coding_latent + rng.normal(0, 7.5), 30, 99), 1)
        teamwork_score = round(clip(69 + 8.5 * soft_latent + rng.normal(0, 7.0), 32, 98), 1)

        # Experiential features
        projects_count = int(np.clip(rng.poisson(2.4 + 0.9 * max(0, coding_latent) + 0.4 * max(0, diligence)), 0, 8))
        internships_count = int(np.clip(rng.binomial(2, clip(0.32 + 0.16 * diligence + 0.10 * coding_latent, 0.05, 0.88)), 0, 3))
        certifications_count = int(np.clip(rng.poisson(1.8 + 0.7 * max(0, diligence)), 0, 7))
        hackathons_count = int(np.clip(rng.poisson(1.3 + 0.8 * max(0, coding_latent) + (0.5 if is_cs_it or is_ds else 0.0)), 0, 7))

        coding_rating = int(np.clip(
            1220 + 230 * coding_latent + (140 if (is_cs_it or is_ds) else -60) + rng.normal(0, 120),
            800, 2350
        ))

        # Domain-specific best technical blend
        if is_core:
            tech_index = 0.45 * core_eng_score + 0.20 * python_score + 0.20 * sql_score + 0.15 * aptitude_score
        elif is_circuital:
            tech_index = 0.32 * core_eng_score + 0.24 * python_score + 0.24 * dsa_score + 0.20 * cloud_score
        elif is_ds:
            tech_index = 0.30 * python_score + 0.30 * ml_score + 0.22 * sql_score + 0.18 * dsa_score
        else:
            tech_index = 0.30 * dsa_score + 0.25 * python_score + 0.23 * web_score + 0.22 * sql_score

        soft_index = 0.40 * communication_score + 0.35 * aptitude_score + 0.25 * teamwork_score
        exp_index = min(100.0, projects_count * 9.0 + internships_count * 22.0 + certifications_count * 5.0 + hackathons_count * 6.5)

        # Composite Readiness Score (0 - 100)
        readiness_raw = (
            0.24 * ((cgpa - 5.0) / 5.0 * 100.0)
            + 0.34 * tech_index
            + 0.20 * soft_index
            + 0.16 * exp_index
            + 0.06 * ((coding_rating - 800) / 1550.0 * 100.0)
            - 6.5 * backlogs
        )
        readiness_score = round(clip(readiness_raw, 12.0, 98.5), 1)
        if readiness_score >= 76:
            readiness_band = "Industry Ready"
        elif readiness_score >= 60:
            readiness_band = "Developing"
        else:
            readiness_band = "Needs Upskilling"

        # Hidden placement probability rule (non-linear + realistic noise)
        z = (
            -1.75
            + 0.62 * (cgpa - 7.0)
            + 0.042 * (tech_index - 62.0)
            + 0.028 * (soft_index - 64.0)
            + 0.52 * internships_count
            + 0.16 * projects_count
            + 0.14 * hackathons_count
            + 0.0014 * (coding_rating - 1300)
            - 0.85 * backlogs
            + (0.35 if branch in ("CSE", "IT", "DS") else (0.12 if branch == "ECE" else -0.15))
            + batch_market_shift[batch] * 4.0
            + rng.normal(0, 0.55)
        )
        true_prob = 1.0 / (1.0 + np.exp(-z))
        placed = int(rng.random() < true_prob)

        # Match placed student to an eligible & skill-aligned role
        placed_company_id = ""
        placed_company = "Unplaced"
        placed_role_id = ""
        placed_role = "None"
        package_lpa = 0.0
        package_category = "Unplaced"
        applications_sent = int(np.clip(rng.poisson(max(1.0, 9 + 3 * diligence)), 2, 24))
        eligible_drives = 0
        shortlisted_count = 0

        # Evaluate eligibility across companies
        role_scores = []
        for _, r in df_roles.iterrows():
            elig = (cgpa >= r["min_cgpa"]) and (backlogs <= r["max_backlogs"])
            pref_list = r["preferred_branches"].split(",")
            branch_match = 1.0 if branch in pref_list else 0.45
            if elig:
                eligible_drives += 1
            req_dict = json.loads(r["required_skills_json"])
            s_vals = []
            for sk, req_v in req_dict.items():
                stu_v = locals()[sk]
                s_vals.append(min(1.15, stu_v / max(req_v, 1)))
            skill_fit = float(np.mean(s_vals)) if s_vals else 0.75
            total_fit = skill_fit * branch_match * (1.0 if elig else 0.25)
            role_scores.append((total_fit, elig, r))

        eligible_drives = min(applications_sent, max(1 if placed else 0, int(eligible_drives * 0.42)))
        shortlisted_count = min(eligible_drives, max(1 if placed else 0, int(eligible_drives * clip(true_prob * 0.85, 0.15, 0.95))))

        if placed == 1:
            role_scores.sort(key=lambda x: x[0], reverse=True)
            elig_candidates = [x for x in role_scores if x[1]]
            pool = elig_candidates[:10] if len(elig_candidates) >= 3 else role_scores[:10]
            weights = np.array([max(0.05, x[0]) ** 3 for x in pool])
            weights = weights / weights.sum()
            chosen_idx = int(rng.choice(len(pool), p=weights))
            _, _, chosen_role = pool[chosen_idx]

            placed_company_id = chosen_role["company_id"]
            placed_company = chosen_role["company_name"]
            placed_role_id = chosen_role["role_id"]
            placed_role = chosen_role["role_title"]

            # Package modulated by skill fit + CGPA + noise
            pkg_noise = rng.normal(0.0, chosen_role["package_lpa"] * 0.06)
            skill_bonus = (readiness_score - 65.0) * 0.035
            package_lpa = round(max(3.6, chosen_role["package_lpa"] + skill_bonus + pkg_noise), 2)

            if package_lpa < 7.0:
                package_category = "Low (<7 LPA)"
            elif package_lpa < 11.5:
                package_category = "Mid (7-11.5 LPA)"
            elif package_lpa < 16.5:
                package_category = "High (11.5-16.5 LPA)"
            else:
                package_category = "Premium (16.5+ LPA)"

        students.append({
            "student_id": sid,
            "name": full_name,
            "gender": gender,
            "batch": batch,
            "branch": branch,
            "branch_name": branch_names[branch],
            "tenth_pct": round(tenth_pct, 1),
            "twelfth_pct": round(twelfth_pct, 1),
            "cgpa": cgpa,
            "backlogs": backlogs,
            "python_score": python_score,
            "sql_score": sql_score,
            "ml_score": ml_score,
            "dsa_score": dsa_score,
            "web_score": web_score,
            "cloud_score": cloud_score,
            "core_eng_score": core_eng_score,
            "communication_score": communication_score,
            "aptitude_score": aptitude_score,
            "teamwork_score": teamwork_score,
            "projects_count": projects_count,
            "internships_count": internships_count,
            "certifications_count": certifications_count,
            "hackathons_count": hackathons_count,
            "coding_rating": coding_rating,
            "tech_index": round(tech_index, 1),
            "soft_index": round(soft_index, 1),
            "exp_index": round(exp_index, 1),
            "readiness_score": readiness_score,
            "readiness_band": readiness_band,
            "applications_sent": applications_sent,
            "eligible_drives": eligible_drives,
            "shortlisted_count": shortlisted_count,
            "placed": placed,
            "placed_company_id": placed_company_id,
            "placed_company": placed_company,
            "placed_role_id": placed_role_id,
            "placed_role": placed_role,
            "package_lpa": package_lpa,
            "package_category": package_category,
        })

    df_students = pd.DataFrame(students)

    # Stratified 80/20 split by (placed + branch)
    strat_key = df_students["branch"] + "_" + df_students["placed"].astype(str)
    train_df, test_df = train_test_split(
        df_students, test_size=0.20, random_state=seed, stratify=strat_key
    )

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    df_students.to_csv(DATA_RAW / "students_full.csv", index=False)
    df_companies.to_csv(DATA_RAW / "companies.csv", index=False)
    df_roles.to_csv(DATA_RAW / "job_roles.csv", index=False)

    train_df.to_csv(DATA_PROC / "train.csv", index=False)
    test_df.to_csv(DATA_PROC / "test.csv", index=False)

    print(f"Generated {len(df_students)} students | Placed: {df_students['placed'].mean()*100:.1f}% | Avg Pkg (Placed): {df_students[df_students['placed']==1]['package_lpa'].mean():.2f} LPA")
    print(f"Companies: {len(df_companies)} | Job Roles: {len(df_roles)} | Train: {len(train_df)} | Test: {len(test_df)}")
    return df_students, df_companies, df_roles, train_df, test_df


if __name__ == "__main__":
    generate_all()
