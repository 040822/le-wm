#!/usr/bin/env bash

set -uo pipefail

# =========================
# 用户配置
# =========================

# 用法: codex-goal-resume.sh <SESSION_ID>
SESSION_ID="${1:-}"

if [[ -z "$SESSION_ID" ]]; then
    echo "Usage: $0 <SESSION_ID>" >&2
    exit 1
fi

RESUME_PROMPT="/goal resume"

CHECK_INTERVAL=180

# 如果设置了 CODEX_HOME，就使用它；
# 否则默认使用 ~/.codex
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
AUTH_FILE="$CODEX_HOME/auth.json"

# ChatGPT Codex usage endpoint
USAGE_URL="https://chatgpt.com/backend-api/wham/usage"


# =========================
# 辅助函数
# =========================

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

format_reset_time() {
    local value="$1"

    # Unix epoch seconds: around 10 digits in the foreseeable future
    if [[ "$value" =~ ^[0-9]{10}$ ]]; then
        date -d "@$value" '+%Y-%m-%d %H:%M:%S' 2>/dev/null ||
            printf '%s' "$value"
        return
    fi

    # Unix epoch milliseconds
    if [[ "$value" =~ ^[0-9]{13}$ ]]; then
        date -d "@$((value / 1000))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null ||
            printf '%s' "$value"
        return
    fi

    # ISO-8601 or anything else date(1) understands
    date -d "$value" '+%Y-%m-%d %H:%M:%S' 2>/dev/null ||
        printf '%s' "$value"
}


# =========================
# 依赖检查
# =========================

for cmd in curl jq codex; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log "ERROR: required command not found: $cmd"
        exit 1
    fi
done


# =========================
# 预检 session 是否存在
# =========================
# 避免白等额度恢复后才发现 ID 无效。

SESSION_FILE="$(
    find \
        "$CODEX_HOME/sessions" \
        "$CODEX_HOME/archived_sessions" \
        -type f -name "*${SESSION_ID}*" 2>/dev/null |
        head -n 1
)"

if [[ -z "$SESSION_FILE" ]]; then
    log "ERROR: no saved session found for ID: $SESSION_ID"
    log "Run 'codex resume --all' to pick a valid session."
    exit 1
fi

log "Session rollout file: $SESSION_FILE"


# =========================
# Session lifetime lock
# =========================
#
# Keep this FD open intentionally.
#
# The lock is inherited by `exec codex` so that the same session cannot
# accidentally be resumed by another watcher while Codex is still running.

if command -v flock >/dev/null 2>&1; then
    LOCK_FILE="${TMPDIR:-/tmp}/codex-goal-resume-${SESSION_ID}.lock"

    exec 9>"$LOCK_FILE"

    if ! flock -n 9; then
        log "ERROR: this session is already being watched or resumed."
        exit 1
    fi
fi


# =========================
# 等待额度恢复
# =========================

log "Watching Codex quota."
log "Session: $SESSION_ID"

while true; do

    # 每次都重新读取 token。
    # 如果 Codex 更新了 auth.json，可以自动使用新 token。
    ACCESS_TOKEN="$(
        jq -er '
            .tokens.access_token
            | select(type == "string" and length > 0)
        ' "$AUTH_FILE" 2>/dev/null
    )"

    if [[ $? -ne 0 || -z "$ACCESS_TOKEN" ]]; then
        log "Cannot read Codex access token; retrying in ${CHECK_INTERVAL}s."
        sleep "$CHECK_INTERVAL"
        continue
    fi


    # 查询当前 Codex quota 状态
    RESPONSE="$(
        printf 'header = "Authorization: Bearer %s"\n' "$ACCESS_TOKEN" |
            curl \
                --config - \
                --fail \
                --silent \
                --show-error \
                --max-time 20 \
                "$USAGE_URL" \
                2>/dev/null
    )"

    if [[ $? -ne 0 || -z "$RESPONSE" ]]; then
        log "Usage request failed; retrying in ${CHECK_INTERVAL}s."
        sleep "$CHECK_INTERVAL"
        continue
    fi


    # fail-closed 三态：
    #
    # true    → 额度恢复，resume
    # false   → 正常额度耗尽，等待
    # unknown → API/schema/解析异常，等待
    #
    # 注意不要用 jq -e 承担业务判断：它会把合法的 false
    # 当作非零退出码，从而和解析失败混淆。
    ALLOWED="$(
        jq -r '
            if (.rate_limit.allowed | type) == "boolean"
            then (.rate_limit.allowed | tostring)
            else "unknown"
            end
        ' <<< "$RESPONSE" 2>/dev/null
    )"

    case "$ALLOWED" in
        true)
            log "Codex quota is available."
            break
            ;;

        false)
            log "Codex quota is currently unavailable."
            ;;

        *)
            log "Cannot determine quota state; retrying in ${CHECK_INTERVAL}s."
            sleep "$CHECK_INTERVAL"
            continue
            ;;
    esac


    # 以下数据只用于显示，不参与真正的启动判断。
    PRIMARY_USED="$(
        jq -r '
            .rate_limit.primary_window.used_percent
            // "N/A"
        ' <<< "$RESPONSE"
    )"

    SECONDARY_USED="$(
        jq -r '
            .rate_limit.secondary_window.used_percent
            // "N/A"
        ' <<< "$RESPONSE"
    )"

    PRIMARY_RESET="$(
        jq -r '
            .rate_limit.primary_window.reset_at
            // empty
        ' <<< "$RESPONSE"
    )"

    SECONDARY_RESET="$(
        jq -r '
            .rate_limit.secondary_window.reset_at
            // empty
        ' <<< "$RESPONSE"
    )"


    log "Quota unavailable: primary=${PRIMARY_USED}%, secondary=${SECONDARY_USED}%."

    if [[ -n "$PRIMARY_RESET" ]]; then
        log "Primary reset:   $(format_reset_time "$PRIMARY_RESET")"
    fi

    if [[ -n "$SECONDARY_RESET" ]]; then
        log "Secondary reset: $(format_reset_time "$SECONDARY_RESET")"
    fi

    log "Retrying in ${CHECK_INTERVAL}s."

    sleep "$CHECK_INTERVAL"

done


# =========================
# 恢复 Goal
# =========================

log "Resuming Codex goal with sandbox + auto-review."

exec codex \
    --approve-for-me \
    resume "$SESSION_ID" \
    "$RESUME_PROMPT"
