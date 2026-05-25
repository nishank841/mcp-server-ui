#!/usr/bin/env python3
"""
MCP Server UI — Chatbot interface for Jira + Kubernetes operations
Powered by Claude API with tool use
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

import anthropic
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

app = FastAPI(title="MCP Server UI")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are a helpful DevOps assistant that can manage Jira tickets and Kubernetes infrastructure.

You can:
- Create and manage Jira epics, stories, tasks, and bugs
- Assign and close Jira issues
- Provision Kubernetes app infrastructure (creates a GitHub PR)
- Inspect Kubernetes resources (pods, services, namespaces, ingress, deployments)

When users ask to create Jira tickets, always ask for the project key if not provided (e.g. SCRUM).
Be concise and action-oriented. After completing an action, show the result clearly."""


# ── kubectl helpers ────────────────────────────────────────────────────────────

def run_kubectl(args: List[str], namespace: Optional[str] = None) -> str:
    cmd = ["kubectl"]
    if namespace:
        cmd.extend(["-n", namespace])
    cmd.extend(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"


def get_namespaces() -> str:
    return run_kubectl(["get", "namespaces", "-o", "wide"])

def get_pods(namespace: Optional[str] = None) -> str:
    return run_kubectl(["get", "pods", "-o", "wide"], namespace)

def get_services(namespace: Optional[str] = None) -> str:
    return run_kubectl(["get", "services", "-o", "wide"], namespace)

def get_ingress(namespace: Optional[str] = None) -> str:
    return run_kubectl(["get", "ingress", "-o", "wide"], namespace)

def get_deployment_status(name: str, namespace: str) -> str:
    return run_kubectl(["get", "deployment", name, "-o", "wide"], namespace)

def get_load_balancer_url(service_name: str, namespace: str) -> str:
    for jsonpath in [
        "{.status.loadBalancer.ingress[*].hostname}",
        "{.status.loadBalancer.ingress[*].ip}",
    ]:
        out = run_kubectl(["get", "service", service_name, f"-o=jsonpath={jsonpath}"], namespace)
        if out and not out.startswith("Error"):
            return f"http://{out}"
    return "Load Balancer URL not available yet"


# ── GitHub provisioning ────────────────────────────────────────────────────────

def provision_app_infrastructure(app_name: str, language: str) -> str:
    try:
        from github import Github, GithubException, Auth as GHAuth
        from datetime import datetime
    except ImportError:
        return "ERROR: PyGithub not installed."

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "ERROR: GITHUB_TOKEN environment variable is not set."

    repo_name = "nishank841/kube-helm"
    branch_name = f"infra/provision-{app_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    image, port = "nginx:latest", 80

    files = {
        f"manifests/namespaces/{app_name}.yaml": f"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: {app_name}\n  labels:\n    language: {language.lower()}\n",
        f"manifests/services/{app_name}-service.yaml": f"apiVersion: v1\nkind: Service\nmetadata:\n  name: {app_name}-svc\n  namespace: {app_name}\nspec:\n  selector:\n    app.kubernetes.io/name: app\n    app.kubernetes.io/instance: {app_name}\n  ports:\n  - port: 80\n    targetPort: {port}\n  type: NodePort\n",
    }

    try:
        gh = Github(auth=GHAuth.Token(token))
        repo = gh.get_repo(repo_name)
        base_sha = repo.get_branch(repo.default_branch).commit.sha
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
        committed = []
        for path, content in files.items():
            try:
                existing = repo.get_contents(path, ref=branch_name)
                repo.update_file(path=path, message=f"chore: update {path}", content=content, sha=existing.sha, branch=branch_name)
            except GithubException:
                repo.create_file(path=path, message=f"feat: add {path}", content=content, branch=branch_name)
            committed.append(path)
        pr = repo.create_pull(
            title=f"feat: provision {language} app infrastructure for {app_name}",
            body=f"## Infrastructure provisioning for `{app_name}`\n\n**Language:** {language}\n\n" + "\n".join(f"- `{p}`" for p in committed),
            head=branch_name, base=repo.default_branch,
        )
        return f"SUCCESS: PR created for '{app_name}' ({language})\nPR URL: {pr.html_url}\nFiles:\n" + "\n".join(f"  - {p}" for p in committed)
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Jira helpers ──────────────────────────────────────────────────────────────

def _jira_auth():
    return HTTPBasicAuth(os.environ.get("JIRA_EMAIL", ""), os.environ.get("JIRA_API_TOKEN", ""))

def _jira_url(path: str) -> str:
    base = os.environ.get("JIRA_URL", "").rstrip("/")
    return f"{base}/rest/api/3/{path.lstrip('/')}"

def _jira_headers():
    return {"Accept": "application/json", "Content-Type": "application/json"}


def jira_get_issue(issue_key: str) -> str:
    r = requests.get(_jira_url(f"issue/{issue_key}"), auth=_jira_auth(), headers=_jira_headers())
    if not r.ok:
        return f"ERROR: {r.status_code} {r.text}"
    d = r.json()
    f = d["fields"]
    assignee = (f.get("assignee") or {}).get("displayName", "Unassigned")
    return f"Key: {d['key']}\nSummary: {f.get('summary')}\nType: {f['issuetype']['name']}\nStatus: {f['status']['name']}\nAssignee: {assignee}"


def jira_create_ticket(project_key: str, summary: str, issue_type: str,
                       description: str, epic_key: str = "", assignee_email: str = "") -> str:
    payload: Dict[str, Any] = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
            "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]},
        }
    }
    if assignee_email:
        search = requests.get(_jira_url(f"user/search?query={assignee_email}"), auth=_jira_auth(), headers=_jira_headers())
        if search.ok and search.json():
            payload["fields"]["assignee"] = {"accountId": search.json()[0]["accountId"]}
    if epic_key:
        payload["fields"]["parent"] = {"key": epic_key}

    r = requests.post(_jira_url("issue"), auth=_jira_auth(), headers=_jira_headers(), json=payload)
    if not r.ok and epic_key:
        payload["fields"].pop("parent", None)
        payload["fields"]["customfield_10014"] = epic_key
        r = requests.post(_jira_url("issue"), auth=_jira_auth(), headers=_jira_headers(), json=payload)
    if not r.ok:
        return f"ERROR: {r.status_code} {r.text}"
    key = r.json()["key"]
    base = os.environ.get("JIRA_URL", "").rstrip("/")
    return f"SUCCESS: Created {issue_type} {key}\nURL: {base}/browse/{key}"


def jira_create_epic(project_key: str, summary: str, description: str) -> str:
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": "Epic"},
            "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]},
        }
    }
    r = requests.post(_jira_url("issue"), auth=_jira_auth(), headers=_jira_headers(), json=payload)
    if not r.ok:
        return f"ERROR: {r.status_code} {r.text}"
    key = r.json()["key"]
    base = os.environ.get("JIRA_URL", "").rstrip("/")
    return f"SUCCESS: Created Epic {key}\nURL: {base}/browse/{key}"


def jira_change_assignee(issue_key: str, assignee_email: str) -> str:
    search = requests.get(_jira_url(f"user/search?query={assignee_email}"), auth=_jira_auth(), headers=_jira_headers())
    if not search.ok or not search.json():
        return f"ERROR: Could not find user with email '{assignee_email}'"
    account_id = search.json()[0]["accountId"]
    r = requests.put(_jira_url(f"issue/{issue_key}/assignee"), auth=_jira_auth(), headers=_jira_headers(), json={"accountId": account_id})
    if r.status_code == 204:
        return f"SUCCESS: {issue_key} assigned to {assignee_email}"
    return f"ERROR: {r.status_code} {r.text}"


def jira_close_issue(issue_key: str) -> str:
    r = requests.get(_jira_url(f"issue/{issue_key}/transitions"), auth=_jira_auth(), headers=_jira_headers())
    if not r.ok:
        return f"ERROR fetching transitions: {r.status_code} {r.text}"
    transitions = r.json().get("transitions", [])
    done_id = next((t["id"] for t in transitions if t["name"].lower() in ("done", "closed", "resolved", "close", "complete")), None)
    if not done_id:
        return f"ERROR: No 'Done/Closed' transition found. Available: {[t['name'] for t in transitions]}"
    r2 = requests.post(_jira_url(f"issue/{issue_key}/transitions"), auth=_jira_auth(), headers=_jira_headers(), json={"transition": {"id": done_id}})
    if r2.status_code == 204:
        return f"SUCCESS: {issue_key} closed/done"
    return f"ERROR: {r2.status_code} {r2.text}"


# ── Claude tool definitions ────────────────────────────────────────────────────

TOOLS = [
    {"name": "jira_create_epic", "description": "Create a new Jira Epic", "input_schema": {"type": "object", "properties": {"project_key": {"type": "string"}, "summary": {"type": "string"}, "description": {"type": "string"}}, "required": ["project_key", "summary", "description"]}},
    {"name": "jira_create_ticket", "description": "Create a Jira Story, Task, or Bug, optionally linked to an epic", "input_schema": {"type": "object", "properties": {"project_key": {"type": "string"}, "summary": {"type": "string"}, "issue_type": {"type": "string", "description": "Story, Task, or Bug"}, "description": {"type": "string"}, "epic_key": {"type": "string"}, "assignee_email": {"type": "string"}}, "required": ["project_key", "summary", "issue_type", "description"]}},
    {"name": "jira_get_issue", "description": "Get details of a Jira issue by key", "input_schema": {"type": "object", "properties": {"issue_key": {"type": "string"}}, "required": ["issue_key"]}},
    {"name": "jira_change_assignee", "description": "Assign a Jira issue to a user by email", "input_schema": {"type": "object", "properties": {"issue_key": {"type": "string"}, "assignee_email": {"type": "string"}}, "required": ["issue_key", "assignee_email"]}},
    {"name": "jira_close_issue", "description": "Close or mark a Jira issue as Done", "input_schema": {"type": "object", "properties": {"issue_key": {"type": "string"}}, "required": ["issue_key"]}},
    {"name": "provision_app_infrastructure", "description": "Create a GitHub PR to provision Kubernetes infrastructure for a new app", "input_schema": {"type": "object", "properties": {"app_name": {"type": "string"}, "language": {"type": "string", "description": "python, nodejs, go, java, ruby, php"}}, "required": ["app_name", "language"]}},
    {"name": "get_namespaces", "description": "List all Kubernetes namespaces", "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_pods", "description": "Get pods in a Kubernetes namespace", "input_schema": {"type": "object", "properties": {"namespace": {"type": "string"}}}},
    {"name": "get_services", "description": "Get services in a Kubernetes namespace", "input_schema": {"type": "object", "properties": {"namespace": {"type": "string"}}}},
    {"name": "get_ingress", "description": "Get ingress resources in a Kubernetes namespace", "input_schema": {"type": "object", "properties": {"namespace": {"type": "string"}}}},
    {"name": "get_deployment_status", "description": "Get deployment status by name and namespace", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "namespace": {"type": "string"}}, "required": ["name", "namespace"]}},
    {"name": "get_load_balancer_url", "description": "Get the Load Balancer URL for a Kubernetes service", "input_schema": {"type": "object", "properties": {"service_name": {"type": "string"}, "namespace": {"type": "string"}}, "required": ["service_name", "namespace"]}},
]

TOOL_DISPATCH = {
    "jira_create_epic":          lambda a: jira_create_epic(a["project_key"], a["summary"], a["description"]),
    "jira_create_ticket":        lambda a: jira_create_ticket(a["project_key"], a["summary"], a["issue_type"], a["description"], a.get("epic_key", ""), a.get("assignee_email", "")),
    "jira_get_issue":            lambda a: jira_get_issue(a["issue_key"]),
    "jira_change_assignee":      lambda a: jira_change_assignee(a["issue_key"], a["assignee_email"]),
    "jira_close_issue":          lambda a: jira_close_issue(a["issue_key"]),
    "provision_app_infrastructure": lambda a: provision_app_infrastructure(a["app_name"], a["language"]),
    "get_namespaces":            lambda _: get_namespaces(),
    "get_pods":                  lambda a: get_pods(a.get("namespace")),
    "get_services":              lambda a: get_services(a.get("namespace")),
    "get_ingress":               lambda a: get_ingress(a.get("namespace")),
    "get_deployment_status":     lambda a: get_deployment_status(a["name"], a["namespace"]),
    "get_load_balancer_url":     lambda a: get_load_balancer_url(a["service_name"], a["namespace"]),
}


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    handler = TOOL_DISPATCH.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(args)
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Chat API ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = req.messages
    tool_calls_log = []

    try:
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                return {"response": text, "tool_calls": tool_calls_log}

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                        tool_calls_log.append({"tool": block.name, "input": block.input, "result": result})

                messages = messages + [
                    {"role": "assistant", "content": [b.model_dump() for b in response.content]},
                    {"role": "user", "content": tool_results},
                ]
            else:
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                return {"response": text, "tool_calls": tool_calls_log}

    except anthropic.AuthenticationError:
        return {"response": "Error: Invalid Anthropic API key. Please check the mcp-anthropic-secret.", "tool_calls": []}
    except anthropic.BadRequestError as e:
        msg = str(e)
        if "credit balance" in msg:
            return {"response": "Error: Anthropic API credit balance is too low. Please top up at console.anthropic.com/settings/billing.", "tool_calls": []}
        return {"response": f"Error: {msg}", "tool_calls": []}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "tool_calls": []}


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Chat UI ────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCP DevOps Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e1e4e8; height: 100vh; display: flex; flex-direction: column; }

  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header .logo { width: 32px; height: 32px; background: linear-gradient(135deg, #4f46e5, #7c3aed); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; }
  header h1 { font-size: 16px; font-weight: 600; color: #f0f6fc; }
  header span { font-size: 12px; color: #8b949e; margin-left: 4px; }

  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }

  .message { display: flex; gap: 12px; max-width: 800px; }
  .message.user { align-self: flex-end; flex-direction: row-reverse; }
  .message.assistant { align-self: flex-start; }

  .avatar { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; }
  .message.user .avatar { background: #4f46e5; color: white; }
  .message.assistant .avatar { background: #161b22; border: 1px solid #30363d; color: #8b949e; }

  .bubble { padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; max-width: 640px; }
  .message.user .bubble { background: #4f46e5; color: white; border-bottom-right-radius: 4px; }
  .message.assistant .bubble { background: #161b22; border: 1px solid #30363d; color: #e1e4e8; border-bottom-left-radius: 4px; white-space: pre-wrap; }

  .tool-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .tool-badge { background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 4px 10px; font-size: 11px; color: #7c3aed; display: flex; align-items: center; gap: 4px; }
  .tool-badge::before { content: "⚡"; font-size: 10px; }

  .typing { display: flex; align-items: center; gap: 6px; color: #8b949e; font-size: 13px; }
  .dots { display: flex; gap: 3px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: #8b949e; animation: bounce 1.2s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }

  #input-area { background: #161b22; border-top: 1px solid #30363d; padding: 16px 24px; }
  .input-row { display: flex; gap: 12px; max-width: 800px; margin: 0 auto; }
  #msg { flex: 1; background: #0f1117; border: 1px solid #30363d; border-radius: 10px; padding: 12px 16px; color: #e1e4e8; font-size: 14px; resize: none; outline: none; font-family: inherit; line-height: 1.5; min-height: 48px; max-height: 160px; }
  #msg:focus { border-color: #4f46e5; }
  #msg::placeholder { color: #484f58; }
  button { background: #4f46e5; color: white; border: none; border-radius: 10px; padding: 12px 20px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.15s; white-space: nowrap; }
  button:hover { background: #4338ca; }
  button:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

  .hint { text-align: center; font-size: 12px; color: #484f58; margin-top: 8px; }

  #chat::-webkit-scrollbar { width: 6px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

  a { color: #7c3aed; }
</style>
</head>
<body>

<header>
  <div class="logo">⚙</div>
  <div>
    <h1>MCP DevOps Assistant <span>powered by Claude</span></h1>
  </div>
</header>

<div id="chat">
  <div class="message assistant">
    <div class="avatar">AI</div>
    <div>
      <div class="bubble">Hi! I can help you manage Jira tickets and Kubernetes infrastructure.

Try saying:
• "Create an epic called Payment Gateway in project SCRUM"
• "Create a story under SCRUM-5: setup API gateway"
• "Assign SCRUM-7 to john@example.com"
• "Close SCRUM-6"
• "Provision a nodejs app called auth-service"
• "Show all pods in the mcp-server namespace"</div>
    </div>
  </div>
</div>

<div id="input-area">
  <div class="input-row">
    <textarea id="msg" placeholder="Ask me to create Jira tickets, manage Kubernetes resources..." rows="1"></textarea>
    <button id="send" onclick="sendMessage()">Send</button>
  </div>
  <div class="hint">Press Enter to send · Shift+Enter for new line</div>
</div>

<script>
const chatEl = document.getElementById('chat');
const msgEl = document.getElementById('msg');
const sendBtn = document.getElementById('send');
let history = [];

msgEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

msgEl.addEventListener('input', () => {
  msgEl.style.height = 'auto';
  msgEl.style.height = Math.min(msgEl.scrollHeight, 160) + 'px';
});

function appendMessage(role, text, toolCalls) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'U' : 'AI';

  const inner = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  inner.appendChild(bubble);

  if (toolCalls && toolCalls.length > 0) {
    const badges = document.createElement('div');
    badges.className = 'tool-badges';
    toolCalls.forEach(tc => {
      const badge = document.createElement('div');
      badge.className = 'tool-badge';
      badge.textContent = tc.tool.replace(/_/g, ' ');
      badges.appendChild(badge);
    });
    inner.appendChild(badges);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(inner);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.id = 'typing';
  wrap.innerHTML = `<div class="avatar">AI</div><div class="bubble typing"><span>Thinking</span><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

async function sendMessage() {
  const text = msgEl.value.trim();
  if (!text || sendBtn.disabled) return;

  appendMessage('user', text);
  history.push({ role: 'user', content: text });

  msgEl.value = '';
  msgEl.style.height = 'auto';
  sendBtn.disabled = true;
  showTyping();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history }),
    });
    const data = await res.json();
    removeTyping();
    appendMessage('assistant', data.response, data.tool_calls);
    history.push({ role: 'assistant', content: data.response });
  } catch (err) {
    removeTyping();
    appendMessage('assistant', 'Error: ' + err.message, []);
  }

  sendBtn.disabled = false;
  msgEl.focus();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
