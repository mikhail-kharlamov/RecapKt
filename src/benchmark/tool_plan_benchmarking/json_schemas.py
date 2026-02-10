PLAN_SCHEMA = {
  "title": "ActionPlan",
  "description": "Schema for the structured action plan output from the simple_algorithms",
  "type": "object",
  "additionalProperties": False,
  "required": ["version", "user_request", "memory_used", "assumptions", "plan_steps"],
  "properties": {
    "version": { "type": "string", "enum": ["1.0"] },
    "user_request": { "type": "string" },
    "context_summary": { "type": "string" },
    "memory_used": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["source", "key", "excerpt"],
        "properties": {
          "source": { "type": "string", "enum": ["session", "memory_bank", "tool_result", "note"] },
          "key": { "type": "string" },
          "excerpt": { "type": "string" }
        }
      }
    },
    "assumptions": {
      "type": "array",
      "items": { "type": "string" }
    },
    "risks": {
      "type": "array",
      "items": { "type": "string" }
    },
    "plan_steps": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "kind", "name", "description", "depends_on"],
        "properties": {
          "id": { "type": "string", "pattern": "^s[0-9]+$" },
          "kind": { "type": "string", "enum": ["tool_call", "final_answer"] },
          "name": { "type": "string" },
          "description": { "type": "string" },
          "depends_on": {
            "type": "array",
            "items": { "type": "string", "pattern": "^s[0-9]+$" }
          },
          "condition": { "type": "string" },
          "expected": { "type": "string" },
          "args": {
            "type": "object",
            "description": "For kind=tool_call only. Omit for other kinds.",
            "additionalProperties": True
          },
          "answer_template": {
            "type": "string",
            "description": "For kind=final_answer only. Omit otherwise."
          },
          "collect_from": {
            "type": "array",
            "description": "For kind=final_answer: step ids whose outputs feed the answer.",
            "items": { "type": "string", "pattern": "^s[0-9]+$" }
          }
        }
      }
    }
  }
}
