/**
 * The target registry: the two pipelines this suite can measure, and what differs between them.
 *
 * A table rather than a dispatcher, for the same reason `evaluators/index.ts` is one: decision 7
 * says the renaming half must not move a number, and routing the case loop through a lookup
 * changes how the run is assembled. run.ts still branches on EVAL_TARGET in one place. What this
 * adds is the answer to the question a reader of two stored runs actually has — are these two
 * numbers comparable, and where are they not?
 *
 * They are not comparable in one specific place, and it is written down here rather than left to
 * be rediscovered: the in-process target retrieves ONCE per case and generates N times from the
 * same chunks, while the service plans and retrieves again on every run. So `recall` means
 * "run 1" on both — deliberately, so the series holds — and `recallEveryRun` is a question only
 * the service can be asked. A stored run from before 2026-09-11 has no `target` field at all,
 * because there was only one; `evals/record.ts` reads that absence as unknown rather than as
 * either target, and `eval:diff` prints it under "read these first".
 */
export type TargetDescriptor = {
    id: "ts" | "python";
    title: string;
    /** The module that drives the case loop for this target. */
    runner: string;
    /** What one "run" of a case costs, in work. The reason the two are not the same measurement. */
    perRun: string;
    /** Metrics this target cannot produce at all, and why. Absent is not zero. */
    cannotMeasure: string[];
    /** How the faithfulness judge gets the chunks, when EVAL_JUDGE=1. */
    judgeSource: string;
};

export const TARGETS: TargetDescriptor[] = [
    {
        id: "ts",
        title: "the pipeline in this process — plannedRetrieve + generateText, the functions the route calls",
        runner: "evals/targets/in-process.ts",
        perRun: "retrieval once per case, then N generations from the same chunks",
        cannotMeasure: [
            "recallEveryRun — it retrieves once, so there is no per-run answer to give",
            "cost — nothing writes a query_log row, so the figure would be an estimate and estimates do not belong in a results file",
            "route — there is no router on this side",
        ],
        judgeSource: "the chunks are in hand while the case runs, so the judge is called inside the loop",
    },
    {
        id: "python",
        title: "the Python agent service over HTTP — the same bytes a visitor's browser gets",
        runner: "evals/targets/agent-run.ts, over the client in evals/targets/agent-service.ts",
        perRun: "the whole pipeline: plan, retrieve, answer. HyDE variance therefore reaches coverage",
        cannotMeasure: [
            "chunk TEXTS from the stream — data-sources carries pages, not content",
        ],
        judgeSource: "after the run, in two hops: thread id to trace id on the query_log row, trace id to the context observation the service wrote",
    },
];
