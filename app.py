import json
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st
from groq import Groq
from pydantic import BaseModel, Field


st.set_page_config(page_title="Financial Life Event Detector", layout="wide")


class LifeEventAnalysis(BaseModel):
    detected_event: str = Field(
        description="The primary life event identified (for example, New Baby, Relocation, Job Change, Marriage, Home Purchase, None)."
    )
    confidence_score: float = Field(description="Confidence value between 0.0 and 1.0.")
    supporting_evidence: List[str] = Field(
        description="Specific merchant clues or transactional behaviors that support the decision."
    )
    recommended_action: str = Field(
        description="The next-best action or banking product recommendation based on the event."
    )


DEFAULT_TRANSACTIONS = pd.DataFrame(
    [
        {"customer_id": "C101", "date": "2026-05-01", "merchant": "Babies 'R' Us", "amount": 145.50, "category": "Retail"},
        {"customer_id": "C101", "date": "2026-05-03", "merchant": "Target (Diapers/Formula)", "amount": 89.20, "category": "Groceries"},
        {"customer_id": "C101", "date": "2026-05-10", "merchant": "BuyBuy Baby", "amount": 320.00, "category": "Retail"},
        {"customer_id": "C102", "date": "2026-05-02", "merchant": "Apex Moving Co.", "amount": 1200.00, "category": "Services"},
        {"customer_id": "C102", "date": "2026-05-04", "merchant": "Home Depot", "amount": 450.00, "category": "Home Improvement"},
        {"customer_id": "C102", "date": "2026-05-05", "merchant": "Zillow Escrow Prem", "amount": 5000.00, "category": "Finance"},
        {"customer_id": "C103", "date": "2026-05-01", "merchant": "ADP Payroll Credits", "amount": -2500.00, "category": "Income"},
        {"customer_id": "C103", "date": "2026-05-15", "merchant": "Stripe *TechCorp NewHire", "amount": -3800.00, "category": "Income"},
    ]
)


EVENT_RULES = [
    {
        "event": "New Baby",
        "keywords": ["baby", "diaper", "formula", "stroller", "crib", "nursery", "infant"],
        "recommended_action": "Offer family savings, education planning, and a rewards card for recurring household spend.",
    },
    {
        "event": "Relocation",
        "keywords": ["moving", "movers", "zillow", "u-haul", "moving co", "apartment", "lease"],
        "recommended_action": "Offer relocation support, new address services, and a low-fee account package.",
    },
    {
        "event": "Home Purchase",
        "keywords": ["escrow", "mortgage", "home depot", "realtor", "title", "closing", "inspection"],
        "recommended_action": "Offer mortgage, home insurance, and homeowner refinancing options.",
    },
    {
        "event": "Job Change",
        "keywords": ["payroll", "newhire", "onboarding", "offer letter", "hr", "adp", "workday", "gusto"],
        "recommended_action": "Offer direct deposit setup, cash-flow tools, and income smoothing products.",
    },
    {
        "event": "Marriage",
        "keywords": ["wedding", "bridal", "engagement", "honeymoon", "venue", "ring"],
        "recommended_action": "Offer joint accounts, shared budgeting, and insurance review services.",
    },
]


def ensure_transactions() -> pd.DataFrame:
    if "transactions" not in st.session_state:
        st.session_state.transactions = DEFAULT_TRANSACTIONS.copy()
    return st.session_state.transactions


def sanitize_transactions(df_cluster: pd.DataFrame) -> list[dict]:
    payload = df_cluster[["date", "merchant", "amount"]].copy()
    payload["date"] = pd.to_datetime(payload["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    payload["date"] = payload["date"].fillna("")
    return payload.to_dict(orient="records")


def heuristic_analysis(df_cluster: pd.DataFrame) -> dict:
    merchant_text = " ".join(df_cluster["merchant"].astype(str).str.lower())
    ordered = df_cluster.sort_values("date")
    evidence: list[str] = []

    for rule in EVENT_RULES:
        matches = [keyword for keyword in rule["keywords"] if keyword in merchant_text]
        if matches:
            for keyword in matches:
                matching_rows = ordered[
                    ordered["merchant"].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
                ]
                evidence.extend(
                    f"{row.date}: {row.merchant} (${row.amount:,.2f})"
                    for row in matching_rows.itertuples(index=False)
                )
            return {
                "detected_event": rule["event"],
                "confidence_score": 0.82,
                "supporting_evidence": evidence[:5],
                "recommended_action": rule["recommended_action"],
            }

    if (ordered["amount"] > 0).sum() >= 2 and (ordered["amount"] < 0).sum() >= 1:
        evidence = [f"{row.date}: {row.merchant} (${row.amount:,.2f})" for row in ordered.itertuples(index=False)]
        return {
            "detected_event": "None",
            "confidence_score": 0.35,
            "supporting_evidence": evidence[:5],
            "recommended_action": "No specific event detected. Offer generic savings, budgeting, and account review products.",
        }

    evidence = [f"{row.date}: {row.merchant} (${row.amount:,.2f})" for row in ordered.itertuples(index=False)]
    return {
        "detected_event": "None",
        "confidence_score": 0.25,
        "supporting_evidence": evidence[:5],
        "recommended_action": "No specific event detected. Offer generic savings, budgeting, and account review products.",
    }


def analyze_transaction_cluster(df_cluster: pd.DataFrame, api_key: str) -> dict:
    if df_cluster.empty:
        return {"error": "No transactions available for analysis."}

    if not api_key:
        return heuristic_analysis(df_cluster)

    try:
        client = Groq(api_key=api_key)
        system_prompt = (
            "You are a secure banking AI that analyzes anonymized transaction flows. "
            "Infer whether the customer is undergoing a major life event. "
            "Use only these event labels: Home Purchase, New Baby, Job Change, Relocation, Marriage, None. "
            "Return only valid JSON matching this schema: "
            f"{json.dumps(LifeEventAnalysis.model_json_schema(), indent=2)}"
        )
        user_prompt = (
            "Analyze this chronological sequence of anonymized transactions and return the best matching event. "
            f"Transactions: {json.dumps(sanitize_transactions(df_cluster), indent=2)}"
        )

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        result = json.loads(completion.choices[0].message.content)
        validated = LifeEventAnalysis(**result)
        return validated.model_dump()
    except Exception as exc:
        return {"error": f"LLM analysis failed: {exc}"}


def format_amount(value: float) -> str:
    return f"${value:,.2f}"


def render_confidence(value: float) -> None:
    confidence = max(0.0, min(1.0, float(value)))
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.0%}")


transactions = ensure_transactions()

st.title("Customer Life Event Detection Platform")
st.caption("Privacy-first transaction analysis with a structured LLM or local heuristic fallback.")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Optional. If omitted, the app uses a local heuristic detector.")
    st.markdown("---")
    st.info(
        "The app only sends sanitized transaction metadata to the model. No customer names, account numbers, or other PII are included."
    )

tab1, tab2 = st.tabs(["Live Sandbox Analytics", "Add Custom Transaction"])

with tab1:
    st.subheader("Select a customer profile to analyze")
    distinct_customers = sorted(transactions["customer_id"].unique())
    selected_customer = st.selectbox("Customer ID", distinct_customers)

    customer_log = transactions[transactions["customer_id"] == selected_customer].copy()
    customer_log["date"] = pd.to_datetime(customer_log["date"], errors="coerce")
    customer_log = customer_log.sort_values(by="date").reset_index(drop=True)
    customer_log["date"] = customer_log["date"].dt.strftime("%Y-%m-%d")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("**Transaction History Cluster**")
        display_df = customer_log.copy()
        display_df["amount"] = display_df["amount"].map(format_amount)
        st.dataframe(display_df, width="stretch", hide_index=True)
        run_analysis = st.button("Run Privacy-Compliant Analysis", type="primary")

    with right_col:
        st.markdown("**Inferred Banking Intelligence**")
        if run_analysis:
            with st.spinner("Analyzing anonymized transaction telemetry..."):
                result = analyze_transaction_cluster(customer_log, groq_api_key)

            if "error" in result:
                st.error(result["error"])
            else:
                st.json(result)
                st.metric(label="Inferred Event", value=result.get("detected_event", "Unknown"))
                render_confidence(result.get("confidence_score", 0.0))
                st.success(f"Suggested strategy: {result.get('recommended_action', 'No recommendation available.')}")
                evidence = result.get("supporting_evidence", [])
                if evidence:
                    st.markdown("**Supporting Evidence**")
                    for item in evidence:
                        st.write(f"- {item}")
        else:
            st.info("Click the analysis button to view the classification and supporting evidence.")

with tab2:
    st.subheader("Simulate a custom transaction sequence")
    with st.form("custom_tx_form", clear_on_submit=True):
        c_id = st.selectbox("Assign to Customer ID", ["C101", "C102", "C103", "C201"])
        tx_date = st.date_input("Transaction Date", value=datetime.utcnow().date())
        merchant_name = st.text_input("Merchant String", placeholder="Stroller Warehouse Inc")
        tx_amount = st.number_input("Amount", value=0.0, help="Use negative values for deposits or income.")
        tx_cat = st.selectbox("Category Grouping", ["Retail", "Groceries", "Services", "Finance", "Income", "Utilities"])

        submit_tx = st.form_submit_button("Inject Transaction Into Local Cache")

        if submit_tx:
            if not merchant_name.strip():
                st.error("Merchant String is required.")
            else:
                new_row = {
                    "customer_id": c_id,
                    "date": str(tx_date),
                    "merchant": merchant_name.strip(),
                    "amount": float(tx_amount),
                    "category": tx_cat,
                }
                st.session_state.transactions = pd.concat(
                    [st.session_state.transactions, pd.DataFrame([new_row])], ignore_index=True
                )
                st.success("Transaction logged successfully. Check the Live Sandbox Analytics tab.")
