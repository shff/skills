type ParseResult<'a, T> = Result<(&'a str, T), (&'a str, ParserError)>;

#[derive(Debug, PartialEq)]
enum ParserError {
    Reserved,
    Choice,
    Eof,
    Tag(&'static str),
    TakeWhile,
    MapRes,
    Expected(&'static str),
    NonAssoc,
}

impl ParserError {
    fn fatal(&self) -> bool {
        matches!(self, ParserError::Expected(_) | ParserError::NonAssoc)
    }
}

fn soft<'a, T>(r: ParseResult<'a, T>) -> Result<Option<(&'a str, T)>, (&'a str, ParserError)> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.1.fatal() => Err(e),
        Err(_) => Ok(None),
    }
}

fn expect<'a, P, R>(p: P, what: &'static str) -> impl Fn(&'a str) -> ParseResult<'a, R>
where
    P: Fn(&'a str) -> ParseResult<'a, R>,
{
    move |i| p(i).map_err(|(at, e)| if e.fatal() { (at, e) } else { (at, ParserError::Expected(what)) })
}

fn peek_not<'a>(s: &'static str) -> impl Fn(&'a str) -> ParseResult<'a, ()> {
    move |i| if i.starts_with(s) { Err((i, ParserError::Tag(s))) } else { Ok((i, ())) }
}

fn tag(tag: &'static str) -> impl Fn(&str) -> ParseResult<&str> {
    move |i| {
        if let Some(prefix) = i.strip_prefix(tag) {
            Ok((prefix, &i[..tag.len()]))
        } else {
            Err((i, ParserError::Tag(tag)))
        }
    }
}

fn value<'a, P, R, V>(p: P, v: V) -> impl Fn(&'a str) -> ParseResult<V>
where
    P: Fn(&'a str) -> ParseResult<R>,
    V: Clone,
{
    move |i| p(i).map(|(i, _)| (i, v.clone()))
}

fn map<'a, P, F, A, B>(p: P, f: F) -> impl Fn(&'a str) -> ParseResult<B>
where
    P: Fn(&'a str) -> ParseResult<A>,
    F: Fn(A) -> B,
{
    move |i| p(i).map(|(i, r)| (i, f(r)))
}

fn mapr<'a, P, F, A, B, E>(p: P, f: F) -> impl Fn(&'a str) -> ParseResult<B>
where
    P: Fn(&'a str) -> ParseResult<A>,
    F: Fn(A) -> Result<B, E>,
{
    move |i| p(i).and_then(|(i, r)| f(r).map(|r| (i, r)).or(Err((i, ParserError::MapRes))))
}

fn opt<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<Option<R>>
where
    P: Fn(&'a str) -> ParseResult<R>,
{
    move |i| Ok(soft(p(i))?.map_or((i, None), |(i, r)| (i, Some(r))))
}

fn pair<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<(X, Y)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, r1)| b(i).map(|(i, r2)| (i, (r1, r2))))
}

fn trio<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<(X, Y, Z)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, x)| b(i).and_then(|(i, y)| c(i).map(|(i, z)| (i, (x, y, z)))))
}

fn right<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<Y>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, _)| b(i))
}

fn left<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<X>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, r1)| b(i).map(|(i, _)| (i, r1)))
}

fn middle<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<Y>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, _)| b(i).and_then(|(i, r2)| c(i).map(|(i, _)| (i, r2))))
}

fn outer<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<(X, Z)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, x)| b(i).and_then(|(i, _)| c(i).map(|(i, z)| (i, (x, z)))))
}

fn choice<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<R>
where
    P: Choice<'a, R>,
{
    move |i| p.choice(i)
}

fn take_while<'a, P>(p: P) -> impl Fn(&'a str) -> ParseResult<'a, &'a str>
where
    P: Fn(u8) -> bool,
{
    move |i| match i.bytes().position(|c| !p(c)).unwrap_or(i.len()) {
        0 => Err((i, ParserError::TakeWhile)),
        x => Ok((&i[x..], &i[..x])),
    }
}

fn take_until(p: &'static str) -> impl Fn(&str) -> ParseResult<&str> {
    move |i| i.find(p).map_or(Ok((i, "")), |x| Ok((&i[x..], &i[..x])))
}

fn capture<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<&'a str>
where
    P: Fn(&'a str) -> ParseResult<R>,
{
    move |i| p(i).map(|(i2, _)| (i2, &i[..(i2.as_ptr() as usize - i.as_ptr() as usize)]))
}

fn reserved<'a, P>(p: P, is_reserved: fn(&str) -> bool) -> impl Fn(&'a str) -> ParseResult<'a, &'a str>
where
    P: Fn(&'a str) -> ParseResult<'a, &'a str>,
{
    move |i| match p(i) {
        Ok((i, r)) if !is_reserved(r) => Ok((i, r)),
        Ok(_) => Err((i, ParserError::Reserved)),
        Err((i, r)) => Err((i, r)),
    }
}

fn many<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<Vec<R>>
where
    P: Fn(&'a str) -> ParseResult<R>,
{
    move |mut i| {
        let mut r = Vec::new();
        while let Some((next_input, next_item)) = soft(p(i))? {
            i = next_input;
            r.push(next_item);
        }
        Ok((i, r))
    }
}

fn skip_many<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<'a, ()>
where
    P: Fn(&'a str) -> ParseResult<'a, R>,
{
    move |mut i| {
        while let Some((next, _)) = soft(p(i))? {
            i = next;
        }
        Ok((i, ()))
    }
}

fn one_of<'a, V: Copy>(table: &'static [(&'static str, V)]) -> impl Fn(&'a str) -> ParseResult<'a, V> {
    move |i| {
        let (rest, _) = inline(i)?;
        for &(s, v) in table {
            if let Some(after) = rest.strip_prefix(s) {
                return Ok((after, v));
            }
        }
        Err((i, ParserError::Choice))
    }
}

fn chain<'a, S, P, R1, R2>(sep: S, p: P) -> impl Fn(&'a str) -> ParseResult<Vec<R2>>
where
    S: Fn(&'a str) -> ParseResult<R1>,
    P: Fn(&'a str) -> ParseResult<R2>,
{
    move |i| {
        let Some((mut i, first)) = soft(p(i))? else { return Ok((i, vec![])) };
        let mut res = vec![first];
        while let Some((next_input, next_item)) = soft(right(&sep, &p)(i))? {
            i = next_input;
            res.push(next_item);
        }
        if let Ok((new_i, _)) = sep(i) {
            i = new_i;
        }
        Ok((i, res))
    }
}

enum Assoc {
    Left,
    Non,
}

fn climb<'a, O, E, P, Q, W, B>(min: u8, operand: P, op: Q, power: W, build: B) -> impl Fn(&'a str) -> ParseResult<'a, E>
where
    O: Copy,
    P: Fn(&'a str, u8) -> ParseResult<'a, E> + Copy,
    Q: Fn(&'a str) -> ParseResult<'a, O> + Copy,
    W: Fn(O) -> (u8, Assoc) + Copy,
    B: Fn(O, E, E) -> E + Copy,
{
    move |i| {
        let (mut i, mut lhs) = operand(i, min)?;
        let mut non_assoc = None;
        while let Ok((rest, o)) = op(i) {
            let (bp, assoc) = power(o);
            if bp < min {
                break;
            }
            if non_assoc == Some(bp) {
                return Err((i, ParserError::NonAssoc));
            }
            let (rest, rhs) = expect(climb(bp + 1, operand, op, power, build), "operand")(rest)?;
            non_assoc = matches!(assoc, Assoc::Non).then_some(bp);
            lhs = build(o, lhs, rhs);
            i = rest;
        }
        Ok((i, lhs))
    }
}

const fn eoi(i: &str) -> ParseResult<'_, &str> {
    if i.is_empty() { Ok((i, "")) } else { Err((i, ParserError::Eof)) }
}

fn token<'a, R>(p: impl Fn(&'a str) -> ParseResult<'a, R>) -> impl Fn(&'a str) -> ParseResult<'a, R> {
    move |i| right(inline, &p)(i)
}

fn comment(i: &str) -> ParseResult<'_, &str> {
    right(tag("#"), take_until("\n"))(i)
}

fn inline(i: &str) -> ParseResult<'_, &str> {
    take_while(|c: u8| c == b' ' || c == b'\t')(i).or(Ok((i, "")))
}

trait Choice<'a, O> {
    fn choice(&self, i: &'a str) -> ParseResult<'a, O>;
}

macro_rules! choice(
    ($($id:ident)+ , $($num:tt)+) => (
        impl<'a, OUT, $($id: Fn(&'a str) -> ParseResult<'a, OUT>),+>
            Choice<'a, OUT> for ( $($id),+ ) {
            fn choice(&self, i: &'a str) -> ParseResult<'a, OUT> {
                Err(("", ParserError::Choice))$(.or_else(|e| if e.1.fatal() { Err(e) } else { self.$num(i) }))*
            }
        }
    );
);

choice!(A B, 0 1);
choice!(A B C, 0 1 2);
choice!(A B C D, 0 1 2 3);
choice!(A B C D E, 0 1 2 3 4);
choice!(A B C D E F, 0 1 2 3 4 5);
choice!(A B C D E F G, 0 1 2 3 4 5 6);
choice!(A B C D E F G H, 0 1 2 3 4 5 6 7);
