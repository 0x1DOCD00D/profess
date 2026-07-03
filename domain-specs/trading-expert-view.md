# Securities Trading Desk — A Practitioner's Account

> Written by a trading-desk analyst, describing how the desk actually works and
> the kinds of statements we make over a trading day. No technology here — just
> the business, the vocabulary, and the conventions we take for granted.

---

## What we do

I sit on an equities trading desk. My day is a continuous stream of activity:
brokers buy and sell securities for our clients, positions move between books,
dividends and corporate actions land, and cash settles. Nothing about the day is
decided in advance — it *happens*, and we record each thing as it happens, in
plain language, the way we'd say it out loud across the desk.

If you listened to us for a morning, you'd hear statements like:

> "Mark bought 300 shares of AAPL at 180 dollars."
> "Mark sold 700 shares of MSFT at 150 dollars."
> "Sarah bought 500 shares of TSLA at 240 dollars."
> "Mark's order for 1,000 shares of NVDA was partially filled — 400 at 905
> dollars."
> "Mark received 1,250 dollars in dividends from AAPL."
> "Acme deposited 50,000 dollars into Mark's book."
> "Sarah transferred 100 shares of TSLA to Mark."
> "Mark shorted 200 shares of NVDA at 900 dollars on Nasdaq."
> "Mark covered 200 shares of NVDA at 880 dollars."
> "We charged 9.50 dollars commission on Mark's AAPL trade."

Each one is a complete, self-contained thing we said happened. Put a morning of
them together and you have the story of the desk.

---

## The people and things we talk about

- **Brokers** — the people who transact. Mark, Sarah. Each one keeps a *book*: the
  running record of their positions and cash.
- **Stocks** — the securities, always by ticker, always uppercase. AAPL, MSFT,
  TSLA, NVDA.
- **Accounts / books** — the ledger where a broker's positions and cash live. A
  broker may run more than one book (e.g. a client book and a proprietary book).
- **Clients** — who we act for. Acme. Trades are often *allocated* to a client.
- **Exchanges** — where a trade executes. Nasdaq, NYSE.
- **Orders** — an *instruction* to trade, which may fill all at once, fill in
  pieces, or not fill at all. An order is not the same as a trade: the trade is
  what actually executed.
- **Options and other derivatives** — contracts we sometimes use to hedge. We
  talk about these less often and less precisely than plain stock.

---

## The things that happen (and what we mean by them)

The verbs that move the book:

- **Bought** — we took on shares and paid cash for them.
- **Sold** — we gave up shares and took in cash.
- **Shorted** — we sold shares we don't own, opening a position that profits if
  the price falls.
- **Covered** — we bought back shares to close a short.
- **Transferred** — shares moved from one broker's book to another's.
- **Received** — cash or shares came in, most often a dividend. *(Note: "received"
  here always means money/shares in — it does not mean an order arrived.)*
- **Deposited / withdrew** — cash moved into or out of a book.
- **Filled / partially filled** — how much of an order actually executed, and at
  what price.

The verbs that *describe* but don't move the book directly:

- **Hedged** — we took an offsetting position to limit risk. We say this loosely;
  the exact mechanics vary.
- **Rated** — an analyst's opinion on a stock (overweight, hold, sell).
- **Cancelled** — an order was pulled before it filled.

We glue these together with ordinary words: *at* a price, *from* a source, *to* a
destination, *into* / *out of* an account, *for* a client, *on* an exchange.

---

## How we measure things, and the conventions we assume

- **Share counts** are whole numbers: 300, 700, 150. We sometimes talk in *lots*
  (round hundreds), but a count is a count.
- **Money** is always in dollars: 180 dollars, 1,250 dollars. A **negative**
  amount is a debit — a cost or an outflow.
- **Prices are per share.** "300 shares at 180 dollars" means 180 *each*, a
  54,000-dollar trade. Everyone on the desk reads it that way without thinking.
- **Rates** show up occasionally as percentages — commissions, dividend yields.
- **A buy and a later sell of the same stock net against each other.** If Mark
  buys 300 AAPL and sells 100, he's long 200. The book always reflects the net.
- **Settlement lags.** A trade executes today but cash settles a day or two later.
  On the desk we usually speak as if the position is real the moment it executes;
  the settlement timing is a back-office concern.

---

## What I want to know at the end

For each broker, I want their **book**:

- **What they hold** — which stocks and how many shares, *net* of everything they
  bought, sold, shorted, covered, and had transferred in or out. Longs are
  positive, shorts negative.
- **Their cash** — the running balance after trades, dividends, deposits,
  withdrawals, and commissions.
- **What dividends came in.**
- **Realized profit and loss** where we can compute it — what we made or lost on
  positions we've closed.
- **A log of everything that happened**, in order, so we can reconstruct how the
  book got to where it is.

Here's the honest reality of the desk: **at any given moment I usually only know
part of the picture, and that's fine.** At 9:31 a.m. the book reflects only the
handful of trades we've seen so far. A broker we haven't heard from yet has an
empty book. A stock nobody's traded today doesn't appear. A position we opened
but haven't closed has no realized P&L yet — it's still *open*, not missing. The
book fills in as the day goes on, and a half-built book mid-morning is exactly as
correct as a complete one at the close.

And some things we say, we don't yet track carefully. When someone says "Mark
hedged NVDA with a put," or "the analyst rated AAPL overweight," that's real and
worth saying — but it doesn't change the holdings or cash I'm watching, so it
sits in the record without moving the numbers. We want it *visible* — someone
reviewing the book should see it was said — but we don't quantify it. We note it;
we don't act on it (yet).

---

## The shape of a typical day, start to finish

1. The desk opens. Books are empty.
2. Orders go out and trades start landing — buys and sells, some filling in full,
   some in pieces — and each broker's holdings and cash start to take shape.
3. Cash events arrive: a client deposit, a dividend, commissions debited.
4. Positions move between brokers; shorts get covered.
5. Some looser statements get made — a hedge, an analyst call, a cancelled order —
   that we record but don't quantify into the numbers.
6. By the close, each broker's book tells the full story of their day: what they
   hold net, what cash they have, what they realized, and the ordered trail of how
   they got there.

That's the desk. Plain statements, made as things happen, adding up to a picture
that's always honest about how much of the day it has seen so far — and always
comfortable holding a few things that were said but not yet turned into numbers.
