"""Layer 5: Point-wise verification gate (`verify`) with partial acceptance."""

from __future__ import annotations

from dataclasses import dataclass

from .edit_log import EditOp, apply_edits_to_line, chapter_heading_prefix_end, is_chapter_heading
from .inline_rules import YUEYI_NOISE_CHAR_RE, DirtyLineTask
from .llm_client import resolve_model_family


@dataclass
class LineVerificationOutcome:
    line_num: int
    edits: list[EditOp]
    review_rows: list[dict]
    rejected_count: int


def verify_line_decisions(
    task: DirtyLineTask,
    primary_del_cids: set[int],
    primary_model: str,
    *,
    secondary_del_cids: set[int] | None = None,
    secondary_model: str | None = None,
    primary_family: str | None = None,
    secondary_family: str | None = None,
    proper_nouns: set[str] | None = None,
) -> LineVerificationOutcome:
    """Verify numbered-candidate deletion decisions point-by-point with partial acceptance.

    Gates:
      G0: candidate id exists in `task.candidates` and is disjoint from keep.
      G1: hard protection — never touch punctuation, headings, or mined proper nouns; exact char check.
      G2: Tier-N approved on primary `del`.
      G3: Tier-C (soft-protected) requires cross-family secondary agreement (`del` in both
          AND distinct model + distinct model family), else retained as `status="review"`
          without blocking Tier-N deletions on the same line.
      G4: replay audit via `apply_edits_to_line` asserts all non-edited spans are byte-identical.
    """
    cand_by_id = {c.cid: c for c in task.candidates}
    edits: list[EditOp] = []
    review_rows: list[dict] = []
    rejected_count = 0

    prefix_end = chapter_heading_prefix_end(task.orig_line, allow_long_dirty=True)
    if prefix_end > 0:
        has_dirty_tail = (
            task.orig_line != task.pre_peeled_line
            or bool(YUEYI_NOISE_CHAR_RE.search(task.orig_line[prefix_end:]))
        )
        if not has_dirty_tail:
            return LineVerificationOutcome(task.line_num, [], [], len(primary_del_cids))
    elif is_chapter_heading(task.orig_line.strip()):
        return LineVerificationOutcome(task.line_num, [], [], len(primary_del_cids))

    p_nouns = proper_nouns or set()
    p_fam = (primary_family or resolve_model_family(primary_model)).strip().lower()
    s_fam = (
        (secondary_family or resolve_model_family(secondary_model)).strip().lower()
        if secondary_model
        else ""
    )
    has_cross_family = bool(
        secondary_del_cids is not None
        and secondary_model
        and secondary_model != primary_model
        and p_fam
        and s_fam
        and p_fam != s_fam
    )

    for cid in sorted(primary_del_cids):
        site = cand_by_id.get(cid)
        if site is None:
            rejected_count += 1
            continue

        # G1: verify exact char in original line & hard protections
        if (
            site.orig_start < 0
            or site.orig_end > len(task.orig_line)
            or site.orig_start < prefix_end
            or task.orig_line[site.orig_start : site.orig_end] != site.char
        ):
            rejected_count += 1
            continue

        # Check not inside a proper noun on pre-peeled text
        lo = max(0, site.orig_start - 5)
        hi = min(len(task.orig_line), site.orig_end + 5)
        win = task.orig_line[lo:hi]
        if any(pn in win and site.char in pn for pn in p_nouns):
            # Double check if site sits inside proper noun on pre-peeled line
            if any(pn in task.pre_peeled_line and site.char in pn for pn in p_nouns):
                # More exact check: locate peeled index
                peeled_idx = None
                for p_i, o_i in enumerate(task.peeled_to_orig_offsets):
                    if o_i == site.orig_start:
                        peeled_idx = p_i
                        break
                if peeled_idx is not None:
                    inside_pn = False
                    for pn in p_nouns:
                        pos = 0
                        while True:
                            hit = task.pre_peeled_line.find(pn, pos)
                            if hit < 0:
                                break
                            if hit <= peeled_idx < hit + len(pn):
                                inside_pn = True
                                break
                            pos = hit + 1
                        if inside_pn:
                            break
                    if inside_pn:
                        rejected_count += 1
                        continue

        # G2 / G3: risk tiering
        if site.tier == "N":
            edits.append(
                EditOp(
                    line=task.line_num,
                    start=site.orig_start,
                    end=site.orig_end,
                    op="del",
                    before=site.char,
                    after="",
                    tier="llm_tier_n",
                    reason=site.reason,
                    source=f"llm:{primary_model}",
                    status="approved",
                    confidence=0.95,
                )
            )
        else:
            # Tier-C requires cross-family secondary model agreement
            if has_cross_family and secondary_del_cids is not None and cid in secondary_del_cids:
                edits.append(
                    EditOp(
                        line=task.line_num,
                        start=site.orig_start,
                        end=site.orig_end,
                        op="del",
                        before=site.char,
                        after="",
                        tier="llm_tier_c",
                        reason="tier_c_cross_family_consensus",
                        source=f"llm:consensus({primary_model}+{secondary_model})",
                        status="approved",
                        confidence=0.92,
                    )
                )
            else:
                disallowed_same_family = bool(
                    secondary_del_cids is not None and secondary_model and not has_cross_family
                )
                rev_reason = (
                    "tier_c_same_family_disallowed"
                    if disallowed_same_family
                    else "tier_c_disagreement_or_unverified"
                )
                row_reason = (
                    "tier_c_same_family_disallowed"
                    if disallowed_same_family
                    else "tier_c_requires_consensus"
                )
                edits.append(
                    EditOp(
                        line=task.line_num,
                        start=site.orig_start,
                        end=site.orig_end,
                        op="del",
                        before=site.char,
                        after="",
                        tier="llm_tier_c",
                        reason=rev_reason,
                        source=f"llm:{primary_model}",
                        status="review",
                        confidence=0.5,
                    )
                )
                review_rows.append(
                    {
                        "line": task.line_num,
                        "cid": cid,
                        "char": site.char,
                        "orig_start": site.orig_start,
                        "reason": row_reason,
                        "original": task.orig_line,
                    }
                )

    # G4: mechanical replay invariant check
    if edits:
        apply_edits_to_line(task.orig_line, edits, include_review=False)
        apply_edits_to_line(task.orig_line, edits, include_review=True)

    return LineVerificationOutcome(
        line_num=task.line_num,
        edits=edits,
        review_rows=review_rows,
        rejected_count=rejected_count,
    )
