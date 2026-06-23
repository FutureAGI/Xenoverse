from __future__ import annotations

import random

PURCHASE_SUCCESS = [
    "Acquired {amount:.1f}g of {name}. The compound is {state} at room temperature and costs {cost:.2f} credits. {toxicity_note}",
    "Purchase complete: {amount:.1f}g of {name} for {cost:.2f} credits. Physical state at ambient conditions: {state}. {toxicity_note}",
]

REACTION_SUCCESS = [
    "After {duration:.0f} seconds at {temp:.1f} C and {pressure:.2f} atm, the reaction proceeded to {conversion:.0%} of equilibrium. Obtained: {products_str}. {notes}",
    "Reaction complete. Under these conditions ({temp:.1f} C, {pressure:.2f} atm), {conversion:.0%} conversion was achieved in {duration:.0f}s. Products: {products_str}. {notes}",
]

REACTION_FAIL = [
    "No discernible reaction observed under the given conditions.",
    "The mixture showed no significant chemical transformation.",
]

TOXICITY_NOTE = {
    (0, 2): "Safety profile appears favorable.",
    (2, 4): "Moderate precautions are recommended during handling.",
    (4, 7): "Warning: this compound exhibits notable toxicity. Appropriate protective equipment is required.",
    (7, 10.01): "HAZARD: Extreme toxicity detected. Handle only in certified containment facilities.",
}

MEDICINAL_HINT = {
    "low": "Biological activity screening shows minimal response.",
    "moderate": "Preliminary assays indicate modest biological activity.",
    "high": "Significant biological activity detected; this compound shows promise.",
    "unknown": "Biological activity undetermined.",
}

HIGH_YIELD_NOTE = "The yield exceeded expectations; conditions appear favorable."
LOW_YIELD_NOTE = "Yield was suboptimal under these conditions; consider adjusting temperature or pressure."
EQUILIBRIUM_NOTE = "Reaction appears to have reached equilibrium within the given time."

PATHWAY_SUMMARY = [
    (
        "Pathway analysis for {target} ({num_steps} step(s)): "
        "estimated yield {yield_g:.3f}g at {efficiency:.1%} mass efficiency. "
        "Total cost {total_cost:.2f} credits ({cost_per_gram} cr/g). "
        "Bottleneck: step {bottleneck} ({bottleneck_conv:.0%} conversion). "
        "Overall atom economy: {atom_economy:.1%}. {efficiency_note}"
    ),
    (
        "{num_steps}-step synthesis of {target} - "
        "projected output {yield_g:.3f}g (efficiency: {efficiency:.1%}). "
        "Estimated expenditure: {total_cost:.2f} credits total, {cost_per_gram} cr/g target. "
        "Rate-limiting step: step {bottleneck} ({bottleneck_conv:.0%} conversion). "
        "{efficiency_note}"
    ),
]

PATHWAY_EFFICIENCY_NOTE = {
    "excellent": "Pathway efficiency is excellent.",
    "good": "Pathway efficiency is good.",
    "moderate": "Pathway efficiency is moderate; consider optimizing conditions.",
    "poor": "Pathway efficiency is poor; significant material is lost in intermediate steps.",
    "very poor": "Pathway efficiency is very poor; most starting material is wasted.",
}

ROUTE_FOUND = [
    "Identified {n} synthesis route(s) to {target}. Shortest route requires {min_steps} step(s) starting from: {m1_list}.",
    "Found {n} path(s) leading to {target}. Minimum {min_steps} reaction step(s) needed; base chemicals: {m1_list}.",
]

ROUTE_NOT_FOUND = [
    "No synthesis route to {target} was found within {max_steps} reaction steps.",
    "The reaction network does not connect {target} to purchasable chemicals within {max_steps} steps.",
]


def _toxicity_note(toxicity: float) -> str:
    for (lo, hi), note in TOXICITY_NOTE.items():
        if lo <= toxicity < hi:
            return note
    return TOXICITY_NOTE[(7, 10.01)]


def _medicinal_hint(medicinal_value: float) -> str:
    if medicinal_value < 1.0:
        return MEDICINAL_HINT["low"]
    if medicinal_value < 3.0:
        return MEDICINAL_HINT["moderate"]
    if medicinal_value >= 3.0:
        return MEDICINAL_HINT["high"]
    return MEDICINAL_HINT["unknown"]


def generate_response(template_type: str, **kwargs) -> str:
    if template_type == "purchase_success":
        kwargs["toxicity_note"] = _toxicity_note(kwargs.get("toxicity", 0.0))
        return random.choice(PURCHASE_SUCCESS).format(**kwargs)

    if template_type == "reaction_success":
        notes_parts = []
        conversion = kwargs.get("conversion", 0.0)
        if kwargs.get("reached_equilibrium", False):
            notes_parts.append(EQUILIBRIUM_NOTE)
        elif conversion > 0.75:
            notes_parts.append(HIGH_YIELD_NOTE)
        elif conversion < 0.3:
            notes_parts.append(LOW_YIELD_NOTE)
        kwargs["notes"] = " ".join(notes_parts)
        return random.choice(REACTION_SUCCESS).format(**kwargs)

    if template_type == "reaction_fail":
        return random.choice(REACTION_FAIL)

    if template_type == "pathway_summary":
        rating = kwargs.get("efficiency_rating", "moderate")
        kwargs["efficiency_note"] = PATHWAY_EFFICIENCY_NOTE.get(rating, "")
        return random.choice(PATHWAY_SUMMARY).format(**kwargs)

    if template_type == "route_found":
        return random.choice(ROUTE_FOUND).format(**kwargs)

    if template_type == "route_not_found":
        return random.choice(ROUTE_NOT_FOUND).format(**kwargs)

    return "Operation completed."
