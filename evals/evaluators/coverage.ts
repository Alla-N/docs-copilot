/**
 * ANSWER COVERAGE — answerable questions the pipeline actually answered.
 *
 * The other half of the tradeoff this harness exists to protect: loosening the prompt to raise
 * coverage must not lower guardrails, and either number moving alone is a regression. Which is
 * why coverage and guardrails are two modules and not one function with a flag — they are
 * scored over disjoint populations and they are read against each other, not together.
 *
 * "Answered" here means the case's VERDICT was PASS, not merely that text came back. A case can
 * answer and still fail on `mustContain` (an intent silently dropped) or `mustNotContain` (an
 * injection that got through), and both of those are coverage failures: the question was
 * answerable and the pipeline did not answer it correctly.
 */
import type { Ratio } from "../record";
import type { GoldenContext } from "./types";

export function evaluateCoverage(ctx: GoldenContext): Ratio {
    const { answerable } = ctx.populations;
    return {
        num: answerable.filter((c) => ctx.byId.get(c.id)!.verdict === "PASS").length,
        den: answerable.length,
    };
}
