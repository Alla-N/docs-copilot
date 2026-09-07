/**
 * One place to read required environment variables.
 *
 * `process.env.X!` is a promise to the compiler, not a check on the deployment: an unset
 * variable becomes the string "undefined" inside a Supabase URL and fails three calls later
 * with a message that names none of this. Read it once, fail with the variable's name.
 */
export function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(
            `Missing ${name}. Set it in .env.local (see .env.example) or in the deployment's environment.`
        );
    }
    return value;
}
