# Copyright 2025 Individual Contributor: Anonymous contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned Supplemental text-tool and survey protocol (standalone, no model-client imports).

Parser/prompt/option definitions copied from Supplemental agents/env_utils.py,
agents/tool_prompt.py and agents/tau_usi/{agent,utils}.py. Schema coercion
matches the local Supplemental tau runtime service. Never imports its RL stack.
"""
import ast
import json
import math
import re
from typing import Any, Mapping

class BadRequest(ValueError):
    pass

def extract_fn_call(text):
    """
    Extract function calls from text. Returns:
    - List of function call dicts if valid format found
    - {'error': 'message'} if wrong format detected
    - None if no function call found

    Function name must be: <function=name>
    Parameters can be: <parameter=name>value</parameter> OR <name>value</name>
    Both parameter formats can be mixed.
    """
    if not text:
        return None
    text = re.split(r"<\[[^\]]+\]>", text)[-1].strip()

    # Only accept <function=name> format for function names
    matches = list(re.finditer(r"(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>", text, re.DOTALL))

    if not matches:
        # Check for incomplete function call
        fn_start = re.search(r"(?m)^[ \t]*<function=([^>]+)>", text)
        if fn_start:
            fn_name = fn_start.group(1)

            # Check for wrong closing tag format: </function=...> instead of </function>
            wrong_close_function = re.search(r"</function=", text)
            if wrong_close_function:
                return {
                    "error": f"""**Tool Call Format Error**

You used `</function=` as a closing tag, but closing tags should NOT have `=` in them.

**Your format (WRONG):**
```
<function={fn_name}>
...
</function={fn_name}>
```

**Correct format:**
```
<function={fn_name}>
...
</function>
```

The closing tag should simply be `</function>` without any `=` or name."""
                }

            # First check for wrong format: <parameter>value</parameter> instead of <parameter=name>value</parameter>
            wrong_param_format = re.findall(r"<parameter>([^<]*)</parameter>", text)
            if wrong_param_format:
                return {
                    "error": f"""**Tool Call Format Error**

You used `<parameter>` without specifying the parameter name. The parameter name must be in the opening tag.

**Your format (WRONG):**
```
<function={fn_name}>
<parameter>{wrong_param_format[0][:50]}...</parameter>
</function>
```

**Correct format:**
```
<function={fn_name}>
<parameter=message>{wrong_param_format[0][:50]}...</parameter>
</function>
```

Please use `<parameter=PARAM_NAME>value</parameter>` format. The parameter name (e.g., `message`, `command`, `path`) must be specified in the opening tag like `<parameter=message>`."""
                }

            open_params = len(re.findall(r"<parameter=[^>]+>", text))
            close_params = len(re.findall(r"</parameter>", text))
            has_close_function = bool(re.search(r"</function>", text))

            if (open_params != close_params) or (not has_close_function):
                return {
                    "error": f"""**Tool Call Format Error**

It looks like you started a tool call but didn't close one or more tags (e.g., missing `</parameter>` and/or `</function>`).

**Your format (WRONG):**
```
<function={fn_name}>
<parameter=query>...
<parameter=topk>10
```

**Correct format:**
```
<function={fn_name}>
<parameter=query>...</parameter>
<parameter=topk>10</parameter>
</function>
```

Please make sure every `<parameter=...>` has a matching `</parameter>`, and every `<function=...>` has a closing `</function>`."""
                }
        return None

    # Check for wrong parameter format and incomplete parameters within matched function calls
    for m in matches:
        fn_body = m.group(2)
        fn_name = m.group(1)

        # First check for wrong format: <parameter>value</parameter> instead of <parameter=name>value</parameter>
        wrong_param_format = re.findall(r"<parameter>([^<]*)</parameter>", fn_body)
        if wrong_param_format:
            preview = wrong_param_format[0][:50].replace("\n", " ")
            return {
                "error": f"""**Tool Call Format Error**

You used `<parameter>` without specifying the parameter name. The parameter name must be in the opening tag.

**Your format (WRONG):**
```
<function={fn_name}>
<parameter>{preview}...</parameter>
</function>
```

**Correct format:**
```
<function={fn_name}>
<parameter=message>{preview}...</parameter>
</function>
```

Please use `<parameter=PARAM_NAME>value</parameter>` format. The parameter name (e.g., `message`, `command`, `path`) must be specified in the opening tag like `<parameter=message>`."""
            }

        # Check for incomplete parameters
        open_params = len(re.findall(r"<parameter=[^>]+>", fn_body))
        close_params = len(re.findall(r"</parameter>", fn_body))
        if open_params != close_params:
            return {
                "error": f"""**Tool Call Format Error**

It looks like you started a tool call but didn't close one or more parameter tags (e.g., missing `</parameter>`).

**Your format (WRONG):**
```
<function={fn_name}>
<parameter=query>...
```

**Correct format:**
```
<function={fn_name}>
<parameter=query>...</parameter>
</function>
```

Please make sure every `<parameter=...>` has a matching `</parameter>`."""
            }

    # Extract parameters - support both <parameter=name>value</parameter> and <name>value</name>
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count("\n", prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]

    results = []
    for m in last:
        fn_body = m.group(2)
        fn_name = m.group(1)

        # Extract standard format parameters: <parameter=name>value</parameter>
        standard_params = dict(re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", fn_body, re.DOTALL))

        # Extract XML-style parameters: <name>value</name> (but exclude 'parameter' and 'function' tags)
        xml_params = re.findall(r"<([a-z_][a-z0-9_]*)>(.*?)</\1>", fn_body, re.DOTALL | re.IGNORECASE)
        xml_params_dict = {}
        for param_name, param_value in xml_params:
            # Skip 'parameter' and 'function' tags (these are structural, not parameters)
            if param_name.lower() not in ["parameter", "function"]:
                xml_params_dict[param_name] = param_value.strip()

        # Merge: standard params take precedence, then XML params
        merged_params = {**xml_params_dict, **standard_params}

        results.append({"name": fn_name, "arguments": merged_params})

    return results

def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ""
    for i, tool in enumerate(tools):
        assert tool["type"] == "function"
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        ret += f"Description: {fn['description']}\n"

        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = param_info.get("type", "string")

                # Get parameter description
                desc = param_info.get("description", "No description provided")

                # Handle enum values if present
                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\nAllowed values: [{enum_values}]"

                ret += f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
        else:
            ret += "No parameters are required for this function.\n"

        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret

TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Parameters must be wrapped with <parameter=key>value</parameter>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

def _parse_container(value: Any, expected_type: type, field_name: str) -> Any:
    if isinstance(value, expected_type):
        return value
    if not isinstance(value, str):
        raise ValueError(f"expected {expected_type.__name__}")

    parsed: Any = None
    json_error: Exception | None = None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        json_error = error
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as literal_error:
            raise ValueError(
                f"{field_name} must be a JSON {expected_type.__name__}"
            ) from (json_error or literal_error)
    if not isinstance(parsed, expected_type):
        raise ValueError(f"expected {expected_type.__name__}")
    return parsed

def coerce_schema_value(value: Any, schema: Mapping[str, Any], field_name: str) -> Any:
    """Restore an XML-extracted string to the type required by JSON Schema."""
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)

    if schema_type == "array":
        result = _parse_container(value, list, field_name)
        item_schema = schema.get("items", {})
        return [
            coerce_schema_value(item, item_schema, f"{field_name}[{index}]")
            for index, item in enumerate(result)
        ]

    if schema_type == "object":
        result = _parse_container(value, dict, field_name)
        properties = schema.get("properties", {})
        return {
            key: coerce_schema_value(item, properties[key], f"{field_name}.{key}")
            if key in properties
            else item
            for key, item in result.items()
        }

    if schema_type == "integer":
        if isinstance(value, bool):
            raise ValueError("boolean is not an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
        converted = int(value)
        if converted != value:
            raise ValueError("value is not an integer")
        return converted

    if schema_type == "number":
        if isinstance(value, bool):
            raise ValueError("boolean is not a number")
        if isinstance(value, (int, float)):
            converted = value
        elif isinstance(value, str):
            try:
                converted = json.loads(value.strip())
            except json.JSONDecodeError:
                converted = float(value.strip())
        else:
            converted = float(value)
        if isinstance(converted, bool) or not isinstance(converted, (int, float)):
            raise ValueError("value is not a number")
        if not math.isfinite(converted):
            raise ValueError("number must be finite")
        # Preserve JSON integer values. TauBench hashes Python values with their
        # representation, so changing 250 to 250.0 can incorrectly score an
        # otherwise exact ground-truth action as zero.
        return converted

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("expected true or false")

    if schema_type == "string" and not isinstance(value, str):
        return str(value)
    return value

def coerce_tool_arguments(
    arguments: Mapping[str, Any], tools_info: list[dict[str, Any]], tool_name: str
) -> dict[str, Any]:
    """Coerce known parameters while leaving unknown parameters to TauBench."""
    properties: Mapping[str, Any] = {}
    for tool_info in tools_info:
        function = tool_info.get("function", {})
        if function.get("name") == tool_name:
            properties = function.get("parameters", {}).get("properties", {})
            break

    converted: dict[str, Any] = {}
    for name, value in arguments.items():
        if name not in properties:
            converted[name] = value
            continue
        try:
            converted[name] = coerce_schema_value(value, properties[name], name)
        except (TypeError, ValueError) as error:
            raise BadRequest(f"invalid argument {name!r} for {tool_name}: {error}") from error
    return converted

FIELD_ORDINAL = {
    "task_success": {
        "No - Task failed": 1,
        "No - Due to a policy issue, which the agent clearly explained": 2,
        "Partially - Some progress": 3,
        "Yes - Task completed": 4,
        "Fully - Exceeded expectations": 5,
    },
    "efficiency": {
        "Very inefficient - Too many steps": 1,
        "Somewhat inefficient": 2,
        "About right": 3,
        "Very efficient": 4,
    },
    "question_amount_preference": {
        "Too many": 1,
        "About right": 2,
        "Too few": 1,
    },
    "answer_effort_time": {"High": 1, "Medium": 2, "Low": 3},
    "human_like": {"No": 1, "Partially": 2, "Yes": 3},
    "interaction_flow": {
        "Not smooth": 1,
        "OK": 2,
        "Smooth": 3,
        "Excellent": 4,
    },
    "overall_score": {
        "1 (Very poor)": 1,
        "2 (Poor)": 2,
        "3 (Acceptable)": 3,
        "4 (Good)": 4,
        "5 (Excellent)": 5,
    },
    "reuse": {
        "Absolutely no": 1,
        "No": 2,
        "Maybe": 3,
        "Yes": 4,
        "Absolutely yes": 5,
    },
}

SURVEY_QUESTION_TEXT = {
    "task_success": "Did the agent successfully complete your task?",
    "efficiency": "How efficient was the agent in completing the task?",
    "question_amount_preference": "How did the number of clarifying questions feel to you?",
    "answer_effort_time": "How much time/effort did it take to answer the agent's clarifying questions?",
    "human_like": "Does the agent feel human-like?",
    "interaction_flow": "How smooth was the overall interaction during clarification?",
    "overall_score": "Overall agent performance score (1-5)",
    "reuse": "If you encounter similar problems in life, would you like to reuse this agent?",
}

def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"<(?:seed:)?think>.*?</(?:seed:)?think>|<(?:seed:)?think>.*\Z", "", raw or "", flags=re.S | re.I).strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidates.append(brace_match.group(0).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None

SURVEY_SCHEMA_REVISION = "tau-usi-official-text-options-missing-rng42-v3"
SURVEY_FIELD_NAMES = {
    "task_success": "task_success", "efficiency": "efficiency",
    "question_amount_preference": "question_amount", "answer_effort_time": "answer_effort",
    "human_like": "human_likeness", "interaction_flow": "interaction_flow",
    "overall_score": "overall", "reuse": "reuse_intent",
}


def survey_prompt() -> str:
    lines = []
    for field, options in FIELD_ORDINAL.items():
        lines.append(f"- {field}: {SURVEY_QUESTION_TEXT.get(field, field)}")
        lines.append(f"  Options: {list(options.keys())}")
    questions = "\n".join(lines)
    return f"""Based on the above conversation, fill out the survey below.

{questions}

Respond ONLY with a JSON object mapping each survey field id to exactly one listed option string.
Do not include any text outside JSON."""


def parse_survey_options(raw: Mapping[str, Any]) -> dict[str, float]:
    """Keep valid answers only. Missing/invalid answers stay absent until scoring."""
    normalized = {}
    for field, options in FIELD_ORDINAL.items():
        value = raw.get(field)
        if isinstance(value, dict):
            value = value.get("answer")
        # Match Supplemental's _structure_survey_answers string handling.
        value = value.strip() if isinstance(value, str) else None
        if value in options:
            lo, hi = min(options.values()), max(options.values())
            normalized[SURVEY_FIELD_NAMES[field]] = (options[value] - lo) / (hi - lo)
    return normalized


def fixture_survey_options(raw: Mapping[str, int]) -> dict[str, str]:
    """Translate old repository-owned integer fixtures, never live model output."""
    return {field: next(option for option, value in options.items()
                        if value - min(options.values()) == raw[SURVEY_FIELD_NAMES[field]])
            for field, options in FIELD_ORDINAL.items()}


def tool_call_text(name: str, arguments: Mapping[str, Any]) -> str:
    lines = [f"<function={name}>"]
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"<parameter={key}>{text}</parameter>")
    return "\n".join([*lines, "</function>"])
