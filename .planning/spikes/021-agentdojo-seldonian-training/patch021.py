"""Robustness patches for the AgentDojo harness with a small local policy (loaded with -ml patch021).

1. A tool call whose ``arguments`` is not valid JSON (small models do this) becomes a call with
   empty arguments instead of an exception that kills the whole suite process.
2. Any other exception inside one episode is caught and scored the way the harness scores its own
   API errors (utility False, security True: conservative for the certificate), with the error
   text recorded, instead of killing the process. Counted and reported by stageA.py as `error`.
"""
import json
import logging

from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite import task_suite as ts

_orig_to_tool_call = openai_llm._openai_to_tool_call


def _tolerant_to_tool_call(tool_call):
    try:
        args = json.loads(tool_call.function.arguments)
        if not isinstance(args, dict):
            args = {}
    except (json.JSONDecodeError, TypeError):
        logging.warning("patch021: unparseable tool arguments for %s: %r", tool_call.function.name,
                        tool_call.function.arguments[:200] if tool_call.function.arguments else None)
        args = {}
    return FunctionCall(function=tool_call.function.name, args=args, id=tool_call.id)


openai_llm._openai_to_tool_call = _tolerant_to_tool_call

_orig_run = ts.TaskSuite.run_task_with_pipeline


def _guarded_run(self, agent_pipeline, user_task, injection_task, injections, *args, **kwargs):
    try:
        return _orig_run(self, agent_pipeline, user_task, injection_task, injections, *args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logging.error("patch021: episode %s/%s failed: %s: %s", user_task.ID,
                      getattr(injection_task, "ID", None), type(e).__name__, str(e)[:300])
        print(f"[patch021] episode error {user_task.ID}/{getattr(injection_task, 'ID', None)}: "
              f"{type(e).__name__}: {str(e)[:300]}", flush=True)
        return False, (injection_task is not None)


ts.TaskSuite.run_task_with_pipeline = _guarded_run
