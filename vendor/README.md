# vendor/

The egglog engine this workspace builds against is the **luminal-ai fork**,
<https://github.com/luminal-ai/egglog>, branch `luminal/subsumed-c2c0f151` —
upstream `c2c0f151` plus one commit, `1bb30831`, adding the atomic
`add_subsumed` write (`Write::add_subsumed` /
`TableAction::lookup_or_insert_subsumed`) the substitution walk needs: an
insert-born-retired row, because a staged insert cannot be re-subsumed through
the public API, and even a one-round live window for a copied retired spelling
re-arms the orbits the `:subsume` termination discipline prevents. The same
addition is proposed upstream alongside egglog-experimental #60.

The fork is the source of truth. There is no local checkout, no patch file and
no `[patch]` section any more: cargo fetches the fork from GitHub like any other
git dependency, so a fresh machine needs nothing but `cargo build`. To move to a
newer upstream egglog, rebase `luminal/subsumed-c2c0f151` onto the new upstream
commit, push it, and bump the `rev` in the three manifests that pin egglog —
`Cargo.toml`, `crates/luminal_reference/Cargo.toml`,
`tests/test_runtime/Cargo.toml` (they must stay identical so the engine types
unify across crates).
