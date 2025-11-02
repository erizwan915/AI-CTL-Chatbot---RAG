from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from myollama import chatbot, load_expanded_chunks, build_index
import json, datetime
import re

# =====================================================
# Confidence / escalation helpers
# =====================================================

def low_retrieval_confidence(distances, threshold: float = 1.1) -> bool:
    """
    Distances are L2 (lower is better). Higher average distance = weaker match.
    threshold ~1.0-1.2 is reasonable for all-MiniLM-L6-v2.
    We use this ONLY for logging, not for changing what we show the student.
    """
    if not distances:
        return True
    avg = sum(distances) / len(distances)
    return avg > threshold


def is_knox_email(text: str) -> bool:
    """
    Very simple check: something@knox.edu
    """
    if not text:
        return False
    return re.match(r"^[A-Za-z0-9._%+-]+@knox\.edu$", text.strip().lower()) is not None


# =====================================================
# Local escalation logging
# =====================================================

def save_escalation_to_file(user_id, question, reply, context, distances, convo,
                            student_email, path="escalations_log.jsonl"):
    """
    Append uncertain/off-topic conversations to a local JSON Lines file.
    Each line is a separate JSON object for easy review later.
    """
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "student_email": student_email,
        "question": question,
        "reply": reply,
        "context": context,
        "distances": distances,
        "conversation": convo,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def load_escalations(path="escalations_log.jsonl"):
    """
    Read all saved escalation records from disk and return them as a list of dicts.
    If the file doesn't exist yet, return an empty list.
    """
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


# =====================================================
# FastAPI setup
# =====================================================

app = FastAPI()

# Load RAG knowledge base once at startup
chunks = load_expanded_chunks("expanded_tutor_chunks.csv")
index, embeddings, chunks, embed_model = build_index(chunks)

# We'll track both messages and student email per user
conversations = {}      # user_id -> message list
student_emails = {}     # user_id -> knox email str or None


# Request/response models
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


# =====================================================
# Chat endpoint
# =====================================================

@app.post("/chat", response_model=ChatResponse)
def get_response(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id

    # 1. First time this user shows up: create their conversation, no email yet
    if user_id not in conversations:
        conversations[user_id] = [{
            "role": "system",
            "content": (
                "You are an administrative assistant for the Knox College Center for Teaching and Learning (CTL). "
                "Your job is ONLY to help Knox students understand:\n"
                "- when and where tutors are available,\n"
                "- what subjects they cover,\n"
                "- how to get help or schedule tutoring.\n\n"

                "CRITICAL BEHAVIOR:\n"
                "- If the question is about tutoring, subjects, specific tutors (for example: 'Vansh Chugh'), availability, "
                "locations, scheduling, appointments, CTL services, academic support, study skills, or Knox classes: "
                "ANSWER it using the provided context.\n"
                "- If the question is clearly NOT academic or CTL-related "
                "(for example: weather, sports scores, celebrities, world news), DO NOT guess. "
                "In that case, reply normally if you can, but then ADD THIS EXACT SENTENCE at the end:\n"
                "\"I'm not sure about that because it's outside CTL's tutoring info. I can get a person to follow up if you want.\"\n\n"

                "FORMAT RULES:\n"
                "- Respond in clear, structured Markdown.\n"
                "- Use bullet points for multiple tutors or schedules.\n"
                "- Bold tutor names.\n"
                "- Put blank lines between different tutors.\n"
                "- Include calendar links when available.\n"
                "- Be warm, encouraging, and student-friendly."
            )
        }]
        student_emails[user_id] = None  # not collected yet

        # Immediately greet and ask for their Knox email before anything else
        intro = (
            "Hi! I'm the CTL tutoring assistant for Knox College. "
            "I can help you find tutors, subjects, and availability.\n\n"
            "Before we get started, what's your Knox email so we can follow up if needed?"
            "\n(Example: yourname@knox.edu)"
        )
        return ChatResponse(response=intro)

    # 2. If we don't have their email yet, check if THIS message is an email
    if student_emails[user_id] is None:
        if is_knox_email(request.message):
            # Save it and confirm
            student_emails[user_id] = request.message.strip()
            confirm_msg = (
                f"Thanks! I've got your email as {student_emails[user_id]}.\n\n"
                "How can I help you today? (For example: 'When is math tutoring available?' "
                "or 'How do I book with Vansh Chugh?')"
            )
            return ChatResponse(response=confirm_msg)
        else:
            # Still no email, keep prompting
            need_email_msg = (
                "I just need your Knox email (like yourname@knox.edu) so we can follow up with you.\n"
                "Could you enter that first?"
            )
            return ChatResponse(response=need_email_msg)

    # 3. We *do* have their email. Now we actually run RAG + LLM.
    reply, distances, context = chatbot(
        request.message,
        conversations[user_id],
        index, chunks, embeddings, embed_model
    )

    # Did the model consider this out of scope?
    bot_flagged_out_of_scope = "outside ctl's tutoring info" in reply.lower()

    # Was retrieval weak? (we still log that but won't always show fallback)
    should_log_low_conf = low_retrieval_confidence(distances, threshold=1.1)

    trigger = bot_flagged_out_of_scope or should_log_low_conf

    if trigger:
        background_tasks.add_task(
            save_escalation_to_file,
            user_id,
            request.message,
            reply,
            context,
            distances,
            conversations[user_id],
            student_emails[user_id],
        )

        if bot_flagged_out_of_scope:
            reply = (
                "I'm not completely sure about that — it may be outside what I know for CTL tutoring. "
                "I’ve flagged this so a staff member can follow up."
            )

    return ChatResponse(response=reply)


# =====================================================
# Serve your chat UI
# =====================================================

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


# =====================================================
# Simple review dashboard
# =====================================================

@app.get("/review")
def review_escalations():
    """
    Admin / debug view.
    Shows escalated (low-confidence or out-of-scope) questions.
    """
    data = load_escalations()

    rows_html = ""
    for item in reversed(data):  # newest first
        ts = item.get("timestamp", "n/a")
        user_id = item.get("user_id", "n/a")
        email = item.get("student_email", "n/a")

        question = item.get("question", "")
        reply = item.get("reply", "")
        context_preview = item.get("context", "")[:300] + "…"
        distances = item.get("distances", [])

        # escape angle brackets so HTML doesn't break
        question = question.replace("<", "&lt;")
        reply = reply.replace("<", "&lt;")
        context_preview = context_preview.replace("<", "&lt;")

        avg_dist = (
            round(sum(distances) / len(distances), 3)
            if distances else "n/a"
        )

        rows_html += f"""
        <tr>
          <td style="vertical-align:top; padding:8px; border-bottom:1px solid #ddd;">
            <div><b>Time:</b> {ts}</div>
            <div><b>User ID:</b> {user_id}</div>
            <div><b>Email:</b> {email}</div>
            <div><b>Avg FAISS Dist:</b> {avg_dist}</div>
          </td>

          <td style="vertical-align:top; padding:8px; border-bottom:1px solid #ddd;">
            <div style="font-weight:600; margin-bottom:4px;">Student asked:</div>
            <div style="white-space:pre-wrap;">{question}</div>

            <div style="font-weight:600; margin:12px 0 4px;">Bot replied:</div>
            <div style="white-space:pre-wrap; color:#444;">{reply}</div>
          </td>

          <td style="vertical-align:top; padding:8px; border-bottom:1px solid #ddd;">
            <div style="font-weight:600; margin-bottom:4px;">RAG context excerpt:</div>
            <div style="font-size:13px; line-height:1.4; color:#555; white-space:pre-wrap;">
              {context_preview}
            </div>
          </td>
        </tr>
        """

    if not rows_html:
        rows_html = """
        <tr>
          <td colspan="3" style="padding:20px; text-align:center; color:#666;">
            No escalations yet 🎉
          </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>CTL Chatbot Escalations</title>
      <style>
        body {{
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f5f5f5;
          padding: 40px;
          color: #222;
        }}

        .card {{
          max-width: 1100px;
          margin: 0 auto;
          background: #fff;
          border-radius: 12px;
          box-shadow: 0 8px 24px rgba(0,0,0,0.08);
          border: 1px solid #ddd;
          overflow: hidden;
        }}

        .header {{
          background: #b30000;
          color: #fff;
          padding: 16px 20px;
          font-size: 18px;
          font-weight: 600;
          letter-spacing: 0.3px;
          border-bottom: 3px solid #fff;
        }}

        table {{
          border-collapse: collapse;
          width: 100%;
          font-size: 14px;
        }}

        th {{
          text-align: left;
          background: #fafafa;
          border-bottom: 1px solid #ddd;
          color: #444;
          font-weight: 600;
          padding: 10px 8px;
          vertical-align: top;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="header">CTL Chatbot – Escalations Review</div>
        <table>
          <thead>
            <tr>
              <th style="width:200px;">Meta</th>
              <th style="width:400px;">Conversation Snapshot</th>
              <th>Retrieved CTL Context (preview)</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
