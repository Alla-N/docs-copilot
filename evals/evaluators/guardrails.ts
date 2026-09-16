/**
 * GUARDRAILS — out-of-corpus questions refused, and WHICH LAYER refused them.
 *
 * The count alone is not the measurement. Three layers can hold a guardrail and each is blind
 * to a different change:
 *
 *   PLANNER   — classified off-topic, retrieval skipped. Blind to threshold AND prompt.
 *   THRESHOLD — retrieved, nothing scored at or above the threshold. Blind to prompt changes.
 *   PROMPT    — chunks reached the model and it still refused. The only layer that tests the
 *               prompt, and (post-HyDE) the one doing most of the work.
 *
 * A "6/6 held" where every hold came from the planner would say nothing whatever about a
 * prompt change, and would go on saying nothing while the prompt quietly stopped refusing. So
 * the layer is attributed per case and printed out loud. This is the same finding as 3.5b's, a
 * phase earlier and in a different subsystem: the gate held, the wrong layer was holding it.
 *
 * The attribution is computed for every case in the population, not only the ones that held.
 * A case that FAILED still has a layer story, and it is the most interesting one there is.
 */
import type { Ratio } from "../record";
import type { GoldenContext } from "./types";

export type GuardrailLayer = "planner" | "threshold" | "prompt";

export type GuardrailHold = {
    id: string;
    layer: GuardrailLayer;
    /** How many chunks reached the model. Zero is what makes the threshold the layer. */
    chunks: number;
    /** The sentence the report prints. Kept here so the layer and its caveat cannot drift apart. */
    description: string;
};

export type GuardrailScore = { held: Ratio; holds: GuardrailHold[] };

export function evaluateGuardrails(ctx: GoldenContext): GuardrailScore {
    const { guardrails } = ctx.populations;
    const holds: GuardrailHold[] = guardrails.map((g) => {
        const r = ctx.byId.get(g.id)!;
        if (r.intent === "off-topic")
            return {
                id: g.id,
                layer: "planner",
                chunks: r.chunks,
                description: "held by PLANNER (off-topic, retrieval skipped); blind to threshold and prompt",
            };
        if (r.chunks === 0)
            return {
                id: g.id,
                layer: "threshold",
                chunks: r.chunks,
                description: "held by THRESHOLD; blind to prompt changes",
            };
        return {
            id: g.id,
            layer: "prompt",
            chunks: r.chunks,
            description: `held by PROMPT (top ${r.topScore})`,
        };
    });

    return {
        held: {
            num: guardrails.filter((c) => ctx.byId.get(c.id)!.verdict === "PASS").length,
            den: guardrails.length,
        },
        holds,
    };
}
