// Example grammar: typed, indentation-based, Python-like language.
// Paste combinators.rs below this, in the same file.

// ---- AST ----

#[derive(Debug, Clone, PartialEq)]
pub enum Type<'a> {
    Name(&'a str),
    List(Box<Type<'a>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field<'a> {
    pub name: &'a str,
    pub ty: Type<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item<'a> {
    Struct { name: &'a str, fields: Vec<Field<'a>> },
    Fn { name: &'a str, params: Vec<Field<'a>>, ret: Option<Type<'a>>, body: Vec<Stmt<'a>> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<'a> {
    Let(&'a str, Type<'a>, Expr<'a>),
    Assign(Expr<'a>, Expr<'a>),
    Expr(Expr<'a>),
    Return(Option<Expr<'a>>),
    If(Vec<(Expr<'a>, Vec<Stmt<'a>>)>, Option<Vec<Stmt<'a>>>),
    While(Expr<'a>, Vec<Stmt<'a>>),
    For(&'a str, Expr<'a>, Vec<Stmt<'a>>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Range,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Int(i64),
    Float(f64),
    Str(&'a str),
    Bool(bool),
    Var(&'a str),
    List(Vec<Expr<'a>>),
    Unary(UnOp, Box<Expr<'a>>),
    Binary(BinOp, Box<Expr<'a>>, Box<Expr<'a>>),
    Call(Box<Expr<'a>>, Vec<Expr<'a>>),
    Field(Box<Expr<'a>>, &'a str),
    Index(Box<Expr<'a>>, Box<Expr<'a>>),
}

// ---- Grammar ----


fn is_keyword(w: &str) -> bool {
    matches!(
        w,
        "fn" | "struct" | "return" | "if" | "elif" | "else" | "while" | "for" | "in" | "and" | "or" | "not" | "true" | "false"
    )
}

pub fn parse(src: &str) -> Result<Vec<Item<'_>>, String> {
    let items = many(right(blank, right(indent(0), item)));
    match left(items, right(blank, expect(eoi, "item")))(src) {
        Ok((_, items)) => Ok(items),
        Err((rest, e)) => {
            let before = &src[..src.len() - rest.len()];
            let line = before.matches('\n').count() + 1;
            let col = before.len() - before.rfind('\n').map_or(0, |n| n + 1) + 1;
            let msg = match e {
                ParserError::Expected(what) => format!("expected {what}"),
                ParserError::NonAssoc => "operator cannot be chained".to_string(),
                e => format!("{e:?}"),
            };
            Err(format!("line {line}, col {col}: {msg}"))
        }
    }
}

fn blank(i: &str) -> ParseResult<'_, ()> {
    skip_many(trio(inline, opt(comment), tag("\n")))(i)
}

fn indent<'a>(n: usize) -> impl Fn(&'a str) -> ParseResult<'a, &'a str> {
    move |i| match inline(i) {
        Ok((rest, sp)) if sp.len() == n => Ok((rest, sp)),
        _ => Err((i, ParserError::Tag("indent"))),
    }
}

fn eol(i: &str) -> ParseResult<'_, &str> {
    right(inline, right(opt(comment), choice((tag("\n"), eoi))))(i)
}

fn block<'a, R, P>(
    parent: usize,
    what: &'static str,
    elem: impl Fn(usize) -> P,
) -> impl Fn(&'a str) -> ParseResult<'a, Vec<R>>
where
    P: Fn(&'a str) -> ParseResult<'a, R>,
{
    move |i| {
        let (_, sp) = right(blank, inline)(i)?;
        let n = sp.len();
        if n <= parent {
            return Err((i, ParserError::Tag("block")));
        }
        let (rest, body) = many(right(blank, right(indent(n), elem(n))))(i)?;
        let (next, sp) = right(blank, inline)(rest)?;
        if !next.is_empty() && sp.len() > parent {
            return Err((next, ParserError::Expected(if sp.len() == n { what } else { "matching indentation" })));
        }
        if body.is_empty() {
            return Err((i, ParserError::Tag("block")));
        }
        Ok((rest, body))
    }
}

fn word(i: &str) -> ParseResult<'_, &str> {
    let (rest, w) = take_while(|c: u8| c.is_ascii_alphanumeric() || c == b'_')(i)?;
    if w.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        Ok((rest, w))
    } else {
        Err((i, ParserError::TakeWhile))
    }
}

fn name(i: &str) -> ParseResult<'_, &str> {
    token(reserved(word, is_keyword))(i)
}

fn keyword<'a>(w: &'static str) -> impl Fn(&'a str) -> ParseResult<'a, &'a str> {
    move |i| {
        let (rest, _) = inline(i)?;
        match rest.strip_prefix(w) {
            Some(after) if !after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') => {
                Ok((after, &rest[..w.len()]))
            }
            _ => Err((i, ParserError::Tag(w))),
        }
    }
}

fn sym<'a>(s: &'static str) -> impl Fn(&'a str) -> ParseResult<'a, &'a str> {
    token(tag(s))
}

fn ty(i: &str) -> ParseResult<'_, Type<'_>> {
    choice((map(middle(sym("["), ty, sym("]")), |t| Type::List(Box::new(t))), map(name, Type::Name)))(i)
}

fn field(i: &str) -> ParseResult<'_, Field<'_>> {
    map(outer(name, expect(sym(":"), "':'"), expect(ty, "type")), |(name, ty)| Field { name, ty })(i)
}

fn end(i: &str) -> ParseResult<'_, &str> {
    expect(eol, "end of line")(i)
}

fn body<'a>(n: usize) -> impl Fn(&'a str) -> ParseResult<'a, Vec<Stmt<'a>>> {
    expect(block(n, "statement", stmt), "indented block")
}

fn item(i: &str) -> ParseResult<'_, Item<'_>> {
    choice((struct_item, fn_item))(i)
}

fn struct_item(i: &str) -> ParseResult<'_, Item<'_>> {
    let name = middle(keyword("struct"), expect(name, "struct name"), end);
    let fields = expect(block(0, "field", |_| left(field, end)), "indented fields");
    map(pair(name, fields), |(name, fields)| Item::Struct { name, fields })(i)
}

fn fn_item(i: &str) -> ParseResult<'_, Item<'_>> {
    let params = middle(expect(sym("("), "'('"), chain(sym(","), field), expect(sym(")"), "')'"));
    let ret = opt(right(sym("->"), expect(ty, "type")));
    let head = left(trio(right(keyword("fn"), expect(name, "function name")), params, ret), end);
    map(pair(head, body(0)), |((name, params, ret), body)| Item::Fn { name, params, ret, body })(i)
}

fn clause<'a>(kw: &'static str, n: usize) -> impl Fn(&'a str) -> ParseResult<'a, (Expr<'a>, Vec<Stmt<'a>>)> {
    move |i| pair(middle(keyword(kw), expect(expr, "condition"), end), body(n))(i)
}

fn at<'a, R>(n: usize, p: impl Fn(&'a str) -> ParseResult<'a, R>) -> impl Fn(&'a str) -> ParseResult<'a, R> {
    move |i| right(blank, right(indent(n), &p))(i)
}

enum Rhs<'a> {
    Let(Type<'a>, Expr<'a>),
    Assign(Expr<'a>),
}

fn stmt<'a>(n: usize) -> impl Fn(&'a str) -> ParseResult<'a, Stmt<'a>> {
    move |i| {
        let ret = middle(keyword("return"), opt(expr), end);
        let tail = opt(at(n, right(right(keyword("else"), end), body(n))));
        let if_ = trio(clause("if", n), many(at(n, clause("elif", n))), tail);
        let var = right(keyword("for"), expect(name, "loop variable"));
        let iter = right(expect(keyword("in"), "'in'"), expect(expr, "iterable"));
        let for_ = pair(left(pair(var, iter), end), body(n));
        let let_ = pair(right(sym(":"), expect(ty, "type")), right(expect(sym("="), "'='"), expect(expr, "expression")));
        let rhs = choice((
            map(let_, |(t, v)| Rhs::Let(t, v)),
            map(right(sym("="), expect(expr, "expression")), Rhs::Assign),
        ));
        let simple = left(pair(expr, opt(rhs)), end);
        choice((
            map(ret, Stmt::Return),
            map(if_, |(first, elifs, tail)| {
                let mut arms = vec![first];
                arms.extend(elifs);
                Stmt::If(arms, tail)
            }),
            map(clause("while", n), |(cond, body)| Stmt::While(cond, body)),
            map(for_, |((var, iter), body)| Stmt::For(var, iter, body)),
            mapr(simple, |(lhs, rhs)| match (lhs, rhs) {
                (Expr::Var(var), Some(Rhs::Let(t, val))) => Ok(Stmt::Let(var, t, val)),
                (_, Some(Rhs::Let(..))) => Err(()),
                (lhs, Some(Rhs::Assign(rhs))) => Ok(Stmt::Assign(lhs, rhs)),
                (lhs, None) => Ok(Stmt::Expr(lhs)),
            }),
        ))(i)
    }
}

fn digits(i: &str) -> ParseResult<'_, &str> {
    take_while(|c: u8| c.is_ascii_digit())(i)
}

fn primary(i: &str) -> ParseResult<'_, Expr<'_>> {
    token(choice((
        mapr(capture(pair(digits, opt(pair(tag("."), digits)))), |s: &str| match s.contains('.') {
            true => s.parse().map(Expr::Float).map_err(|_| ()),
            false => s.parse().map(Expr::Int).map_err(|_| ()),
        }),
        mapr(word, |w| match w {
            "true" => Ok(Expr::Bool(true)),
            "false" => Ok(Expr::Bool(false)),
            w if is_keyword(w) => Err(()),
            w => Ok(Expr::Var(w)),
        }),
        map(middle(tag("\""), take_until("\""), expect(tag("\""), "closing '\"'")), Expr::Str),
        map(middle(tag("["), chain(sym(","), expr), expect(sym("]"), "']'")), Expr::List),
        middle(tag("("), expect(expr, "expression"), expect(sym(")"), "')'")),
    )))(i)
}

enum Suffix<'a> {
    Call(Vec<Expr<'a>>),
    Field(&'a str),
    Index(Expr<'a>),
}

fn postfix(i: &str) -> ParseResult<'_, Expr<'_>> {
    let suffix = choice((
        map(middle(sym("("), chain(sym(","), expr), expect(sym(")"), "')'")), Suffix::Call),
        map(right(left(sym("."), peek_not(".")), expect(name, "field name")), Suffix::Field),
        map(middle(sym("["), expect(expr, "index"), expect(sym("]"), "']'")), Suffix::Index),
    ));
    map(pair(primary, many(suffix)), |(e, sufs)| {
        sufs.into_iter().fold(e, |e, s| match s {
            Suffix::Call(args) => Expr::Call(Box::new(e), args),
            Suffix::Field(f) => Expr::Field(Box::new(e), f),
            Suffix::Index(ix) => Expr::Index(Box::new(e), Box::new(ix)),
        })
    })(i)
}

const NOT: u8 = 3;
const NEG: u8 = 8;

fn power(op: BinOp) -> (u8, Assoc) {
    match op {
        BinOp::Or => (1, Assoc::Left),
        BinOp::And => (2, Assoc::Left),
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => (4, Assoc::Left),
        BinOp::Range => (5, Assoc::Non),
        BinOp::Add | BinOp::Sub => (6, Assoc::Left),
        BinOp::Mul | BinOp::Div | BinOp::Mod => (7, Assoc::Left),
    }
}

fn build<'a>(op: BinOp, a: Expr<'a>, b: Expr<'a>) -> Expr<'a> {
    Expr::Binary(op, Box::new(a), Box::new(b))
}

fn binop(i: &str) -> ParseResult<'_, BinOp> {
    let op = one_of(&[
        ("==", BinOp::Eq),
        ("!=", BinOp::Ne),
        ("<=", BinOp::Le),
        (">=", BinOp::Ge),
        ("<", BinOp::Lt),
        (">", BinOp::Gt),
        ("..", BinOp::Range),
        ("+", BinOp::Add),
        ("-", BinOp::Sub),
        ("*", BinOp::Mul),
        ("/", BinOp::Div),
        ("%", BinOp::Mod),
    ]);
    choice((left(op, peek_not(">")), value(keyword("and"), BinOp::And), value(keyword("or"), BinOp::Or)))(i)
}

fn operand(i: &str, min: u8) -> ParseResult<'_, Expr<'_>> {
    let not = map(right(keyword("not"), expect(|i| expr_at(i, NOT), "operand")), |e| Expr::Unary(UnOp::Not, Box::new(e)));
    let neg = map(right(sym("-"), expect(|i| expr_at(i, NEG), "operand")), |e| Expr::Unary(UnOp::Neg, Box::new(e)));
    if min <= NOT { choice((not, neg, postfix))(i) } else { choice((neg, postfix))(i) }
}

fn expr_at(i: &str, min: u8) -> ParseResult<'_, Expr<'_>> {
    climb(min, operand, binop, power, build)(i)
}

fn expr(i: &str) -> ParseResult<'_, Expr<'_>> {
    expr_at(i, 0)
}

