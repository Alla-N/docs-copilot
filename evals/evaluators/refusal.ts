/**
 * FALSE REFUSALS — the expected page was retrieved and the answer refused anyway.
 *
 * This gets its own line because recall is PAGE-level. `expectedSource` names a page, so ANY
 * chunk of it counts as "retrieved". A case can therefore report recall 12/12 and still refuse,
 * because the chunk that survived carried none of the answer. `changed-7` did exactly that on
 * Day 15: the migration page's intro chunk ("use the command below to add the migration skill")
 * made the top 5 and none of the substantive ones did, so the model correctly refused a
 * question the corpus answers.
 *
 * That is the most expensive failure this project can have — the whole point is grounded
 * answers, not silence — and reporting it only as `FAIL` hid the cause for a day. Two metrics
 * that are both green can still be hiding the thing between them.
 *
 * The rule differs by target, and the difference is not cosmetic:
 *   one retrieval per case  → "the expected page was retrieved, and the model refused"
 *   retrieval on every run  → "the model refused every time, and the page was in front of it
 *                             in at least one of those runs"
 */
import type { GoldenContext } from "./types";
import type { EvalCase } from "../datasets/types";

/**
 * Empty in the retrieval-only tier (EVAL_RUNS=0): nothing generated an answer, so nothing
 * refused, and a zero there would read as "no false refusals" rather than "not measured".
 */
export function evaluateFalseRefusals(ctx: GoldenContext): EvalCase[] {
    if (!ctx.generated) return [];
    return ctx.populations.answerable.filter((c) => {
        const r = ctx.byId.get(c.id)!;
        if (r.answered !== 0) return false;
        return ctx.perRunRetrieval ? (r.foundRuns ?? 0) > 0 : r.retrieved === "yes";
    });
}
