/**
 * Who counts toward which denominator — in ONE place.
 *
 * See the comment on `Populations` in ./types.ts for why each exclusion is an exclusion. This
 * module is the only thing in the suite allowed to decide it.
 */
import type { EvalCase } from "../datasets/types";
import type { Populations, Scoreboard } from "./types";

/**
 * A case "ran" if it produced a verdict about the pipeline at all. An id with no result is
 * treated as not having run rather than as a silent pass: the alternative is `byId.get(id)!`
 * throwing in the middle of scoring, which is how a missing case became a crash rather than a
 * number in an earlier version of this loop.
 */
export const ran = (byId: Scoreboard) => (c: EvalCase): boolean => {
    const r = byId.get(c.id);
    return r !== undefined && r.verdict !== "ERROR";
};

/**
 * Split the active cases into the four populations the report and the record both read.
 *
 * Note that `parked` is NOT filtered by `ran`. A parked case that could not run is still parked,
 * and the "an expectFail case is now PASSING" check has to see every one of them or a fix could
 * hide behind an outage.
 */
export function partition(active: EvalCase[], byId: Scoreboard): Populations {
    const didRun = ran(byId);
    return {
        answerable: active.filter((c) => c.shouldAnswer === true && !c.injection && !c.expectFail && didRun(c)),
        guardrails: active.filter((c) => c.shouldAnswer === false && !c.injection && !c.expectFail && didRun(c)),
        injections: active.filter((c) => c.injection && !c.expectFail && didRun(c)),
        parked: active.filter((c) => c.expectFail),
    };
}
