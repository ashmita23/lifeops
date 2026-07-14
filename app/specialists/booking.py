"""Booking specialist: makes a restaurant reservation.

This is the human-in-the-loop demo. book_reservation is a high-stakes action,
so it's registered as an approval-required tool (see _requires_approval in
app/agent.py): the agent proposing it PAUSES and surfaces the details -
including the guest name used for ID verification - for the user to approve
before anything is booked, exactly like the destructive-delete gate. The
booking itself is mocked (no real reservation API is freely available); the
point being demonstrated is the approval/verification flow, not the vendor
integration.
"""

import hashlib


def book_reservation(args: dict, raw_text: str) -> dict:
    """Execute an (approved) reservation and return a confirmation. Only ever
    runs AFTER the approval gate - the agent cannot call this directly without
    the user confirming the pending action first."""
    restaurant = args.get("restaurant") or "Unknown"
    when = args.get("datetime") or ""
    party_size = args.get("party_size")
    guest_name = args.get("guest_name") or "Unnamed"

    # Deterministic confirmation code (hash, not random) so it's testable.
    digest = hashlib.sha1(f"{restaurant}|{when}|{guest_name}".encode()).hexdigest()[:6].upper()

    return {
        "status": "booked",
        "restaurant": restaurant,
        "datetime": when,
        "party_size": party_size,
        "guest_name": guest_name,
        "confirmation_number": f"LO-{digest}",
    }
