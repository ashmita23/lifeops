"""Golden dataset for the LifeOps agent - a fixed set of test cases with
known-good expectations, run against the real model (see run_evals.py) so
you get a repeatable pass/fail + latency report instead of manually
retyping the same prompts and eyeballing the result each time.

Each case:
- id: short unique name
- turns: user messages sent in sequence on the same session (most cases
  are one turn; delete-confirmation needs two)
- expected_tools: tool names expected in the FINAL turn's result.actions
  (order-insensitive exact set match) - use this OR expected_min_actions,
  not both, depending on whether exact tool identity is predictable
- expected_min_actions: for cases where the model's exact tool choices
  aren't fully predictable (e.g. the 5-call cap), just check the action
  COUNT meets a minimum instead of exact tool names
- expected_keywords: optional substrings (case-insensitive) expected
  somewhere in the final turn's reply text
"""

GOLDEN_CASES = [
    {
        "id": "simple_reminder",
        "turns": ["remind me to call the accountant tomorrow at 2pm"],
        "expected_tools": ["create_reminder"],
        "expected_keywords": ["accountant"],
    },
    {
        "id": "explicit_calendar_event_only",
        "turns": ["schedule a calendar event called Budget Review tomorrow at 3pm"],
        "expected_tools": ["create-event"],
        "expected_keywords": ["budget review"],
    },
    {
        "id": "explicit_reminder_and_calendar_both",
        "turns": [
            "remind me to call mom tomorrow at 5pm and also put it on my calendar"
        ],
        "expected_tools": ["create_reminder", "create-event"],
        "expected_keywords": None,
    },
    {
        "id": "multi_step_chain",
        "turns": [
            "check my reminders, add a reminder to buy milk tomorrow at 9am, "
            "and schedule a lunch event with sam tomorrow at noon"
        ],
        # Not requiring a specific read tool for "check my reminders" -
        # get_daily_summary and list_reminders are both legitimate choices.
        # The two writes are what actually matter for this case.
        "expected_tools": ["create_reminder", "create-event"],
        "expected_keywords": None,
    },
    {
        "id": "delete_requires_confirmation",
        "turns": [
            "schedule a calendar event called Eval Test Meeting tomorrow at 1pm",
            "delete my eval test meeting",
            "yes, delete it",
        ],
        "expected_tools": ["delete-event"],
        "expected_keywords": ["deleted"],
    },
    {
        "id": "cap_hit_stops_at_five",
        "turns": [
            "check my reminders, check my calendar events, check my journal "
            "entries, add a reminder to buy milk tomorrow at 9am, add a reminder "
            "to call the dentist tomorrow at 10am, and schedule a lunch event "
            "with sam tomorrow at noon"
        ],
        "expected_tools": None,
        "expected_min_actions": 5,
        "expected_keywords": None,
    },
    {
        "id": "mcp_fuzzy_calendar_search",
        "turns": [
            "schedule a calendar event called Weekly Sync Eval tomorrow at 4pm",
            "cancel my weekly sync eval meeting tomorrow",
            "yes, cancel it",
        ],
        "expected_tools": ["delete-event"],
        "expected_keywords": ["cancel", "weekly sync eval"],
    },
    {
        # Regression case for a real reported bug: the agent scheduled an
        # event, then claimed "no other meetings scheduled for the entire
        # day" in a later turn - flatly contradicting the event it had just
        # created itself. Requiring the follow-up answer to actually
        # mention "lunch" forces it to be grounded in a real list-tool
        # result for the full scope asked about, not an extrapolated guess.
        "id": "availability_answer_reflects_real_events",
        "turns": [
            "schedule a lunch event with sam tomorrow at noon",
            "what does my calendar look like tomorrow?",
        ],
        "expected_tools": None,
        "expected_keywords": ["lunch"],
    },
    {
        # Regression case for the other half of the same real bug: a single
        # message asking for two things (create + delete) had its delete
        # half silently dropped once the create half needed a confirmation
        # round-trip. Checking the delete actually lands by the final turn
        # confirms it isn't forgotten once the conversation moves on.
        "id": "multi_part_request_delete_not_dropped",
        "turns": [
            "remind me to call mom tomorrow at 5pm",
            "schedule a lunch event with sam tomorrow at noon and also delete my call mom reminder",
            "yes, delete it",
        ],
        "expected_tools": ["delete_reminder"],
        "expected_keywords": None,
    },
]
