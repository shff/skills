---
name: rust-combinator-parsers
description: How to write fast parser combinators in Rust with good error messages (closure-based, &str input, no external crate). Use when writing or optimizing a hand-rolled combinator parser in Rust, adding operator precedence to one, or fixing poor parse errors from backtracking combinators.
---

# Rust combinator parsers

Reference implementation, in this skill's directory:

- `combinators.rs`: the generic combinators, from `type ParseResult` down. Copy it as a base.
- `example.rs`: an example grammar (typed, indentation-based, Python-like language) built on `combinators.rs`. To use it, paste `combinators.rs` below it in the same file.

Benchmarked against a hand-written lexer + recursive descent: it ends up within ~10% of the hand-written parser, with exact line and column errors.

## Core shape

```rust
type ParseResult<'a, T> = Result<(&'a str, T), (&'a str, ParserError)>;

fn pair<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<'a, (X, Y)>
where A: Fn(&'a str) -> ParseResult<'a, X>, B: Fn(&'a str) -> ParseResult<'a, Y>,
{ move |i| a(i).and_then(|(i, x)| b(i).map(|(i, y)| (i, (x, y)))) }
```

- The output is zero-copy: `&'a str` slices go into the AST.
- The building blocks are `tag`, `map`, `mapr`, `value`, `opt`, `pair`, `trio`, `left`, `right`, `middle`, `outer`, `many`, `chain` (separated list), `choice` (a trait implemented for tuples by a macro), `take_while`, `take_until`, `capture` and `token` (skip spaces, then run `p`).
- Factory functions that take a `&'static str` need an explicit `'a`: `fn sym<'a>(s: &'static str) -> impl Fn(&'a str) -> ParseResult<'a, &'a str>`. If you elide it, you get the error "implementation of `Fn` is not general enough".

## Performance rules, in order of impact

1. **No allocation on failure.** Error variants hold `&'static str` or nothing, never `String`. A backtracking parser fails all the time: the `String` version was 3× slower.
2. **Precedence climbing, not one function per level.** With a level per precedence, every operand passes through every level, and each level retries its operators. Use a single operator parser, a `power(op) -> (u8, Assoc)` table and a generic `climb`. Removing the levels took the parser from 1.6× to 1.08× the hand-written time.
3. **Fail on the first byte.** `keyword(w)` is `inline`, then `strip_prefix(w)`, then a check that the next character is not alphanumeric or `_`. Do not lex a whole word and then compare it. Check reserved words with `matches!`, not `slice.contains`. `take_while` should take `Fn(u8) -> bool` and use `bytes().position`, which is safe for ASCII predicates.
4. **Scan once.** Parse `digits (. digits)?` once and decide int or float from the result. Parse `word` once and match on `true`, `false`, keywords or a name. Merge statements that share a prefix: parse `expr`, then optionally `: type = …` or `= …`.
5. These had no measurable effect: byte dispatch instead of a small `choice`, avoiding rescans of blank lines, and `one_of` tables instead of a `choice` of `sym`s (keep those only if they make the code clearer).
6. The style of the code costs nothing. A grammar written entirely as `map(pair(...))` runs as fast as sequential `let (i, x) = p(i)?;`.
7. Use Criterion with `harness = false` and assert equal ASTs before timing. Single `Instant` runs gave ±8% noise. Compare ratios within one run, because machine state can change absolute times by 2×.

## Recursion

Recursive parsers must recurse through `fn` items, which are zero-sized types. Never store a closure that contains itself.

```rust
fn stmt<'a>(n: usize) -> impl Fn(&'a str) -> ParseResult<'a, Stmt<'a>> { move |i| choice((..., block(n, stmt) ...))(i) }
fn expr_at(i: &str, min: u8) -> ParseResult<'_, Expr<'_>> { climb(min, operand, binop, power, build)(i) }
```

Calling a function that returns `impl Fn` from inside its own closure body works, because the closure is built at call time and not stored.

## Precedence climbing as a generic combinator

```rust
enum Assoc { Left, Non }

fn climb<'a, O, E, P, Q, W, B>(min: u8, operand: P, op: Q, power: W, build: B) -> impl Fn(&'a str) -> ParseResult<'a, E>
where O: Copy, P: Fn(&'a str, u8) -> ParseResult<'a, E> + Copy, Q: Fn(&'a str) -> ParseResult<'a, O> + Copy,
      W: Fn(O) -> (u8, Assoc) + Copy, B: Fn(O, E, E) -> E + Copy,
{
    move |i| {
        let (mut i, mut lhs) = operand(i, min)?;
        let mut non_assoc = None;
        while let Ok((rest, o)) = op(i) {
            let (bp, assoc) = power(o);
            if bp < min { break; }
            if non_assoc == Some(bp) { return Err((i, ParserError::NonAssoc)); }
            let (rest, rhs) = expect(climb(bp + 1, operand, op, power, build), "operand")(rest)?;
            non_assoc = matches!(assoc, Assoc::Non).then_some(bp);
            lhs = build(o, lhs, rhs);
            i = rest;
        }
        Ok((i, lhs))
    }
}
```

- Prefix operators belong in the grammar's `operand(i, min)`. Their operand is `expr_at(i, PREFIX_BP)`.
- To limit where a low-precedence prefix such as `not` can appear, accept it only when `min <= NOT_BP`.
- If the operator parser can match the start of a longer token, guard it: `-` must not match the start of `->`, so use `left(op, peek_not(">"))`.

## Error messages

Plain backtracking combinators give useless errors such as `line 1: Eof`. `many`, `opt` and `choice` swallow failures, so the outermost `eoi` reports instead.

The fix: once the parser is past a point of no return, a failure is final. Nothing backtracks over it, and it is reported as it is. Parser literature calls this "commit" or "cut".

- Give the error type fatal variants, `Expected(&'static str)` and `NonAssoc`, and a `fatal()` method.
- `expect(p, "what")` turns a recoverable failure of `p` into `Expected(what)` at the position of the failure. A fatal error passes through unchanged.
- Recovering combinators use a single helper, so `fatal()` is checked in only three places (`soft`, `expect`, `choice!`):

```rust
fn soft<'a, T>(r: ParseResult<'a, T>) -> Result<Option<(&'a str, T)>, (&'a str, ParserError)> {
    match r { Ok(v) => Ok(Some(v)), Err(e) if e.1.fatal() => Err(e), Err(_) => Ok(None) }
}
// many:  while let Some((n, x)) = soft(p(i))? { i = n; r.push(x); }  Ok((i, r))
// opt:   Ok(soft(p(i))?.map_or((i, None), |(i, r)| (i, Some(r))))
// choice!: .or_else(|e| if e.1.fatal() { Err(e) } else { self.N(i) })
```

- Put `expect` after each point of no return: after a keyword (header, end of line, block), after an opening bracket (the closing bracket), after a binary or prefix operator (the operand), and on the end of line of a simple statement.
- Do not put `expect` where another construct shares the prefix. For example, the `.` of a field access must not commit when the input is `..`; use `left(sym("."), peek_not("."))`.
- Compute line and column from `src.len() - rest.len()`.
- With an indentation-based syntax: after `many(stmt)` in a block, if the next non-blank line is indented deeper than the parent, report "expected statement" when the indent is equal to the block's and "matching indentation" when it is not.
- Cost: about 3% speed, because only the failure path changes.
- Test it with a table of invalid inputs. Assert that each one is rejected, and print the messages with `--nocapture`.

## Indentation-sensitive grammars without a lexer

- `blank` skips lines that are whitespace or comments only.
- `indent(n)` requires exactly `n` leading spaces.
- `eol` is `inline`, an optional comment, then `\n` or end of input.
- `block(parent, what, elem)` reads the width `n` of the next line, requires `n > parent`, then runs `many(blank, indent(n), elem(n))`.
- Blocks that follow, such as `elif` and `else`, use `at(n, p) = right(blank, right(indent(n), p))`.
- Simple statements consume their own `eol`. Compound statements consume the `eol` of their header and then the block.
