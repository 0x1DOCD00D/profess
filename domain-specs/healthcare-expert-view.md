# Medication Administration — A Practitioner's Account

> Written by a clinical nurse-informatics analyst, describing how a patient's
> medication course actually unfolds on the ward and the kinds of statements we
> make as care happens. No technology here — just the clinical reality, the
> vocabulary, and the conventions we take for granted. Scope is medication
> administration, not all of patient care.

---

## What we do

On the ward, a patient's medications are not planned out as one finished document.
They *unfold*. A doctor orders a drug. Nurses give doses over hours and days. The
pharmacy sends stock up. Doses get held, missed, or refused. An order gets stopped
when it's no longer needed. Each of these is something we state plainly, as it
happens, and record on the patient's chart.

If you followed a patient through a course of treatment, you'd hear statements
like:

> "Dr. Chen prescribed 500 mg of amoxicillin for patient 4821, by mouth, every 8
> hours for 7 days."
> "Nurse Alvarez gave 500 mg of amoxicillin to patient 4821 by mouth."
> "Nurse Alvarez gave another 500 mg of amoxicillin to patient 4821 by mouth."
> "Dr. Chen prescribed 10 mg of morphine for patient 4821, every 4 hours as
> needed for pain."
> "Nurse Osei gave 10 mg of morphine to patient 4821 by IV."
> "The 2 p.m. dose of amoxicillin for patient 4821 was held — patient was in
> surgery."
> "Patient 4821 refused the morphine."
> "Pharmacy dispensed 14 doses of amoxicillin to ward 3 West."
> "Dr. Chen recorded a penicillin allergy for patient 4821."
> "Dr. Chen discontinued the amoxicillin for patient 4821."

Each one is a complete, self-contained thing that happened. Put the course
together and you have the patient's medication story.

---

## The people and things we talk about

- **Patients** — identified by a chart number, e.g. 4821, **never by name** in our
  records. This is deliberate and non-negotiable.
- **Drugs** — medications by name. Amoxicillin, morphine. Some are **controlled
  substances** (like morphine) and carry extra rules around witnessing and waste.
- **Doctors / prescribers** — who orders the medication. Dr. Chen.
- **Nurses** — who actually give the doses. Alvarez, Osei.
- **Pharmacists** — who dispense stock and check for interactions.
- **Routes** — how a dose is given: by mouth (oral), IV, intramuscular (IM),
  subcutaneous, topical.
- **Wards** — the units stock is sent to. 3 West.
- **Allergies** — what a patient must not be given, ideally with the reaction.
- **Symptoms and observations** — things the patient reports or we notice. We
  mention these constantly, though they sit outside the strict medication record.

---

## The things that happen (and what we mean by them)

The verbs that build the record:

- **Prescribed** — a doctor opened an order: this drug, this dose, this route,
  this often, for this long. An order is an *intention*.
- **Administered (gave)** — a nurse actually gave a dose. This is the real event,
  and it is **deliberately separate** from the order. The most important
  distinction in this whole domain is *ordered* vs *actually given*.
- **Held** — a scheduled dose was deliberately not given (patient in surgery,
  vitals out of range). The order stands; this dose was skipped on purpose.
- **Refused** — the patient declined the dose.
- **Discontinued** — an active order was stopped before its course ended.
- **Dispensed** — the pharmacy released stock to a ward.
- **Recorded** — we noted something important on the chart, like an allergy.

The verbs that *describe* but don't slot into the formal order/dose record:

- **Titrated** — a dose was adjusted up or down over time. We say this often; the
  specifics vary by drug and patient.
- **Reported / noted** — the patient told us something, or we observed it
  (nausea, pain level, "resting comfortably").

We glue these together with ordinary words: *of* a drug, *to* or *for* a patient,
*by* / *via* a route, *every* so many hours, *for* so many days, *as needed* for a
reason.

---

## How we measure things, and the conventions we assume

- **Doses** are amounts with units: 500 mg, 10 mg, 5 ml. Some drugs are dosed by
  the patient's weight — milligrams per kilogram — and the actual amount is worked
  out per patient.
- **Frequency** is a time interval: every 8 hours, every 4 hours. **"As needed"
  (PRN)** is different from a fixed schedule — the dose is given only when a
  condition is met, so a PRN order may legitimately have *no* doses given against
  it.
- **Duration** is a span of days: for 7 days.
- **Counts** show up when pharmacy sends stock: 14 doses.
- **An order is a ceiling, not a guarantee.** "Every 8 hours for 7 days" describes
  what's *allowed*; what was actually given is a separate tally that may be fewer
  (held, refused, discontinued early) and should never silently exceed the order.
- **Doses accumulate; orders don't.** A second administration of amoxicillin is a
  second line in the dose record, not a replacement of the first. But a second
  *prescription* of the same drug usually supersedes or amends the first.

---

## What I want to know at the end

For each patient, I want their **medication record**:

- **What's currently ordered** — each active drug, with its dose, route,
  frequency, duration, and who prescribed it.
- **What's actually been given** — the list of doses administered, with the drug,
  amount, route, and which nurse gave it. This is separate from what was ordered:
  an order is an intention; an administration is what really happened. *If I can
  only have one of these two, I want what was actually given.*
- **What was held or refused** — so a gap in dosing is explained, not mysterious.
- **What's been stopped.**
- **Known allergies** — checked against what's ordered.

Here's the clinical reality: **the chart is almost always partial, and that's
completely normal.** A drug that was prescribed an hour ago but hasn't been given
yet has no doses recorded against it — it's simply pending its first dose, not an
error. A PRN order may go a whole shift with nothing given. A patient we've only
just admitted has a nearly empty record. The chart fills in as care happens, and a
half-built chart mid-shift is exactly as correct as a full one at discharge.

And a great deal of what we say doesn't fit neatly into the medication record at
all. "Patient 4821 reported nausea." "Dr. Chen titrated the morphine." "Patient
resting comfortably." These matter clinically and we say them all the time — but
they don't slot into the structured list of orders and doses. We want them
*visible* — anyone reading the chart should at least see that they *were* said —
even when they're not captured as a formal order or dose.

---

## The shape of a course, start to finish

1. A patient is admitted; their medication record is nearly empty. We reconcile
   any allergies and home medications first.
2. A doctor prescribes — orders start to populate, each one still with no doses
   given against it yet.
3. Nurses administer — the record of *what actually happened* starts to build,
   dose by dose, separate from the orders. Some scheduled doses are held or
   refused, and we say so.
4. Pharmacy dispenses stock; allergies and observations get recorded along the
   way.
5. Orders get discontinued or adjusted as treatment changes.
6. Throughout, plenty gets said that doesn't fit the formal record — symptoms,
   titrations, bedside notes — and that's expected.

That's the ward. Plain statements, made as care happens, adding up to a record
that's always honest about how much of the course it has seen so far — that keeps
the *ordered* and the *actually given* firmly apart — and that's always
comfortable holding a great deal that was said but never meant to fit a tidy
structure.
